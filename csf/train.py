"""
Code for training models unsupervised using contrastive sensor fusion.
"""

import tensorflow as tf
from absl import flags, logging

import csf.data
import csf.global_flags as gf
import csf.utils
from csf.encoder import RESNET_REPRESENTATION_LAYERS, resnet_encoder

FLAGS = flags.FLAGS


# Required hyperparameters
flags.DEFINE_integer(
    "model_tilesize",
    None,
    "Tilesize model accepts for unsupervised learning. "
    "Views are asymmetrically cropped to this size from `data_tilesize` (see data.py).",
    lower_bound=1,
)
flags.DEFINE_float("learning_rate", None, "Learning rate for unsupervised training.")
flags.DEFINE_float("band_dropout_rate", None, "Final rate of dropping out bands.")
flags.DEFINE_list(
    "layer_loss_weights",
    None,
    "Weights for loss at various layers, as a comma-separated list of name:weight "
    "pairs like `conv4_block5_out:0.5`.",
)

# Optional hyperparameters, with sensible defaults
flags.DEFINE_integer(
    "learning_rate_warmup_batches",
    None,
    "How many batches to warm up learning rate over. "
    "If unspecified, learning rate warmup is not used.",
)
flags.DEFINE_integer(
    "band_dropout_rate_warmup_batches",
    None,
    "How many batches to increase band dropout rate over. "
    "If unspecified, band dropout is constant.",
)
flags.DEFINE_float(
    "softmax_temperature", 0.1, "Temperature to use for softmax loss.", lower_bound=0.01
)

flags.DEFINE_float(
    "random_hue_delta",
    None,
    "Maximum amount to randomize hue between views.",
    lower_bound=0.0,
    upper_bound=1.0,
)
flags.DEFINE_float(
    "random_saturation_delta",
    None,
    "Maximum amount to randomize saturation between views.",
    lower_bound=0.0,
    upper_bound=1.0,
)
flags.DEFINE_float(
    "random_brightness_delta",
    None,
    "Maximum amount to randomize brightness between views.",
    lower_bound=0.0,
    upper_bound=1.0,
)
flags.DEFINE_float(
    "random_contrast_delta",
    None,
    "Maximum amount to randomize contrast between views.",
    lower_bound=0.0,
    upper_bound=1.0,
)

# Non-model training settings
flags.DEFINE_string(
    "out_dir", None, "Path used to store the outputs of unsupervised training."
)
flags.DEFINE_integer(
    "train_batches", None, "Number of batches of unsupervised training to carry out."
)
flags.DEFINE_string(
    "initial_checkpoint",
    None,
    "Path to a checkpoint used to continue unsupervised training from."
    "If unspecified, initialize model weights randomly.",
)
flags.DEFINE_integer(
    "summary_frequency", 100, "How many batches to wait between saving summaries."
)
flags.DEFINE_integer(
    "checkpoint_frequency", 500, "How many batches to wait between saving checkpoints."
)

flags.mark_flags_as_required(
    [
        "model_tilesize",
        "learning_rate",
        "band_dropout_rate",
        "layer_loss_weights",
        "train_batches",
        "out_dir",
    ]
)


@flags.validator("layer_loss_weights")
def _check_layer_loss_weights(values):
    try:
        _parse_layer_loss_weights(values)
    except Exception:
        return False
    return True


def _parse_layer_loss_weights(values):
    """
    Internal function used to access and validate layer loss weights.
    Do not use directly.
    """
    result = {}
    for value in values:
        name, weight = value.split(":")
        assert name in RESNET_REPRESENTATION_LAYERS
        result[name] = float(weight)
    return result


def layer_loss_weights():
    return _parse_layer_loss_weights(FLAGS.layer_loss_weights)


def input_shape():
    return (FLAGS.batch_size, FLAGS.model_tilesize, FLAGS.model_tilesize, gf.n_bands())


def _dropout_rate(step):
    with tf.name_scope("schedule_band_dropout_rate"):
        return csf.utils.optional_warmup(
            step, FLAGS.band_dropout_rate, FLAGS.band_dropout_rate_warmup_batches
        )


def _learning_rate(step):
    with tf.name_scope("schedule_learning_rate"):
        return csf.utils.optional_warmup(
            step, FLAGS.learning_rate, FLAGS.learning_rate_warmup_batches
        )


def _create_view(scene, step):
    """
    Apply augmentation to a set of input imagery, creating a new view.

    Parameters
    ----------
    scene : tf.Tensor
        A tensor of aligned input imagery.
    step : tf.Tensor
        A scalar, integer Tensor holding the current step.

    Returns
    -------
    tf.Tensor
        A view of the input imagery with crop, band dropout, and jitter applied.
    """
    if FLAGS.model_tilesize != FLAGS.data_tilesize:
        scene = tf.image.random_crop(scene, input_shape(), name="asymmetric_crop")

    scene = tf.nn.dropout(
        scene,
        _dropout_rate(step),
        noise_shape=(FLAGS.batch_size, 1, 1, gf.n_bands()),
        name="band_dropout",
    )

    if FLAGS.random_hue_delta:
        scene = tf.image.random_hue(scene, FLAGS.random_hue_delta)
    if FLAGS.random_saturation_delta:
        scene = tf.image.random_saturation(
            scene,
            1.0 - FLAGS.random_saturation_delta,
            1.0 + FLAGS.random_saturation_delta,
        )
    if FLAGS.random_brightness_delta:
        scene = tf.image.random_brightness(scene, FLAGS.random_brightness_delta)
    if FLAGS.random_contrast_delta:
        scene = tf.image.random_contrast(
            scene, 1.0 - FLAGS.random_contrast_delta, 1.0 + FLAGS.random_contrast_delta
        )

    return scene


def _contrastive_loss(representation_1, representation_2):
    with tf.name_scope("contrastive_loss"):
        representation_1 = tf.reshape(representation_1, (FLAGS.batch_size, -1))
        representation_2 = tf.reshape(representation_2, (FLAGS.batch_size, -1))

        # Element [i, j] is the dot-product similarity of the i-th representation of
        # view 1 and the j-th representation of view 2 for scenes (i, j) in the batch.
        # The diagonal contains the similarities of matching scenes.
        similarities = tf.linalg.matmul(
            representation_1, representation_2, transpose_b=True, name="similarities"
        )
        tf.summary.histogram(
            "similarities_histogram",
            similarities,
            description="Histogram of similarities between views for each pair of "
            "scenes in the batch.",
        )
        tf.summary.image(
            "similarities_matrix",
            tf.expand_dims(tf.expand_dims(similarities, axis=0), axis=-1),
            description="Matrix of similarities between views for each pair of "
            "scenes in the batch.",
        )

        # Rescale similarities to apply softmax temperature
        similarities = tf.divide(
            similarities, FLAGS.softmax_temperature, name="sharpened_similarities"
        )

        with tf.name_scope("forward"):  # Predict view 2 from view 1
            softmax = tf.nn.log_softmax(similarities, axis=1, name="log_probabilities")
            nce_loss_forward = tf.negative(
                tf.reduce_mean(tf.linalg.diag_part(softmax)), name="nce_loss_forward"
            )

        with tf.name_scope("backward"):  # Predict view 1 from view 2
            softmax = tf.nn.log_softmax(similarities, axis=0, name="log_probabilities")
            nce_loss_backward = tf.negative(
                tf.reduce_mean(tf.linalg.diag_part(softmax)), name="nce_loss_backward"
            )

        nce_loss_total = tf.add(
            nce_loss_forward, nce_loss_backward, name="nce_loss_total"
        )
        tf.summary.scalar(
            "nce_loss_total",
            nce_loss_total,
            description="Sum of forward and backward NCE losses.",
        )

        with tf.name_scope("compute_accuracy"):
            # Ideal predictions mean the greatest logit for each view is paired
            # (i.e. the diagonal dominates each row and column).
            ideal_predictions = tf.range(
                0, FLAGS.batch_size, 1, dtype=tf.int64, name="ideal_predictions"
            )
            predictions = tf.argmax(similarities, name="predictions")
            correct_predictions = tf.cast(tf.equal(predictions, ideal_predictions))
            batch_accuracy = tf.reduce_mean(correct_predictions)
        tf.summary.scalar(
            "batch_accuracy",
            batch_accuracy,
            description="Fraction of scenes in the batch matched correctly.",
        )

    return nce_loss_total


# TODO(Aidan): re-enable autograph conversion
#  @tf.function
def _train_step(batch, encoder, optimizer, step):
    """
    Perform the gradient computation and application for a single batch of training.

    Parameters
    ----------
    batch : tf.Tensor
        A batch of input imagery.
    encoder : tf.keras.Model
        A ResNet-style encoder producing representations at various layers.
    optimizer : tf.optimizers.Optimizer
        The optimizer used for training.
    step : int or tf.Tensor
        Which step of training this is.

    Returns
    -------
    tf.Tensor
        The loss for this step.
    """
    view_1 = _create_view(batch, step)
    view_2 = _create_view(batch, step)
    layers_and_weights = layer_loss_weights().items()
    losses = []

    with tf.GradientTape() as tape:
        representations_1 = encoder(view_1)
        representations_2 = encoder(view_2)

        for layer, weight in layers_and_weights:
            with tf.name_scope("layer_{}".format(layer)):
                loss = _contrastive_loss(
                    representations_1[layer], representations_2[layer]
                )
                losses.append(weight * loss)

        loss_total = tf.reduce_sum(losses, name="loss_total")

    tf.summary.scalar(
        "loss_total", loss_total, description="Total loss for all layers."
    )

    gradients = tape.gradient(loss_total, encoder.trainable_weights)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_weights))

    return loss_total


def train_unsupervised():
    # TODO(Aidan): TPU integration
    # TODO(Aidan): checkpoint saving and restoring
    # TODO(Aidan): implement summary frequency

    logging.info(
        "Starting unsupervised training with flags:\n{}".format(
            FLAGS.flags_into_string()
        )
    )

    logging.debug("Building global objects.")
    summary_writer = tf.summary.create_file_writer(FLAGS.out_dir)
    # TODO(Aidan): determine if this is necessary
    #  tf.summary.record_if(not gf.using_tpu())

    step = tf.Variable(
        0,
        trainable=False,
        name="step",
        dtype=tf.dtypes.int32,
        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
    )
    tf.summary.experimental.set_step(step)
    learning_rate = _learning_rate(step)

    logging.debug("Loading dataset.")
    ds = csf.data.load_dataset()

    logging.debug("Building model and optimizer.")
    encoder = resnet_encoder(input_shape()[1:])
    optimizer = tf.optimizers.Adam(learning_rate, clipnorm=1.0)
    # TODO(Aidan): parameterize optimizers and kwargs

    logging.info("Beginning unsupervised training.")
    with summary_writer.as_default():
        tf.summary.scalar("learning_rate", learning_rate, description="Learning rate.")
        for batch in enumerate(ds):
            _train_step(batch, encoder, optimizer, step)

            if step.assign_add(1) >= FLAGS.train_batches:
                break

    logging.info("Done with unsupervised training.")
