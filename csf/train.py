"""
Code for training models unsupervised using contrastive sensor fusion.

Some components based on https://github.com/tensorflow/models/tree/master/official/bert.
"""

import tensorflow as tf
from absl import flags, logging

import csf.data
import csf.distribution
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

# Optional hyperparameters, with sensible defaults.
# For best performance, do tune some of these.
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
flags.DEFINE_float(
    "softmax_temperature",
    0.1,
    "Temperature to use for softmax loss. Changing this parameter is not recommended.",
    lower_bound=0.01,
)
flags.DEFINE_float(
    "gradient_clipnorm",
    1.0,
    "Clip gradients with norm above this value. "
    "Changing this parameter is not recommended.",
)

# Non-model training settings
flags.DEFINE_string(
    "out_dir", None, "Path used to store the outputs of unsupervised training."
)
flags.DEFINE_string(
    "initial_checkpoint",
    None,
    "Path to a checkpoint used to continue unsupervised training from."
    "If unspecified, initialize model weights randomly.",
)
flags.DEFINE_integer(
    "callback_frequency",
    100,
    "How many batches to train for between writing summaries and checkpoints, and "
    "updating schedules.",
)
flags.DEFINE_integer("max_checkpoints", 100, "The maximum number of checkpoints kept.")
flags.DEFINE_integer(
    "keep_checkpoint_every_n_hours",
    None,
    "Every `n` hours, marks a checkpoint to be kept permanently. "
    "If left unspecified, disables this behavior.",
)
flags.mark_flags_as_required(
    [
        "model_tilesize",
        "learning_rate",
        "band_dropout_rate",
        "layer_loss_weights",
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
    return (
        csf.distribution.replica_batch_size(),
        FLAGS.model_tilesize,
        FLAGS.model_tilesize,
        gf.n_bands(),
    )


def _create_view(scene, seed=None):
    """
    Apply augmentation to a set of input imagery, creating a new view.
    Note that this function is autograph-traced and takes a Python integer input (seed),
    so keep the number of calls with distinct seeds to a minimum.
    Do not pass Python values to any other argument.

    Parameters
    ----------
    scene : tf.Tensor
        A tensor of aligned input imagery.
    dropout_rate : tf.Tensor
        A scalar, float Tensor holding the current dropout rate.
        Included as an argument to work well with scheduling and autograph.
    seed : int, optional
        Random seed to use. Used to ensure that views get different random numbers.

    Returns
    -------
    tf.Tensor
        A view of the input imagery with crop, band dropout, and jitter applied.
    """
    seed = seed or FLAGS.random_seed

    if FLAGS.model_tilesize != FLAGS.data_tilesize:
        scene = tf.image.random_crop(scene, input_shape(), name="crop", seed=seed)
    if FLAGS.random_brightness_delta:
        scene = tf.image.random_brightness(
            scene, FLAGS.random_brightness_delta, seed=seed
        )
    if FLAGS.random_contrast_delta:
        scene = tf.image.random_contrast(
            scene,
            1.0 - FLAGS.random_contrast_delta,
            1.0 + FLAGS.random_contrast_delta,
            seed=seed,
        )
    scene = tf.nn.dropout(
        scene,
        0.66,  # TODO(Aidan: schedule dropout
        noise_shape=(csf.distribution.replica_batch_size(), 1, 1, gf.n_bands()),
        name="band_dropout",
        seed=seed,
    )

    return scene


def _contrastive_loss(representation_1, representation_2):
    """
    Compute the contrastive loss for a pair of representations.

    Parameters
    ----------
    representation_1 : tf.Tensor
        The representations for view 1 over this batch.
    representation_2 : tf.Tensor
        The representations for view 2 over this batch.

    Returns
    -------
    (tf.Tensor, tf.Tensor)
        The total loss and accuracy over this batch.
    """
    flat_1 = tf.reshape(representation_1, (csf.distribution.replica_batch_size(), -1))
    flat_2 = tf.reshape(representation_2, (csf.distribution.replica_batch_size(), -1))

    # Element [i, j] is the dot-product similarity of the i-th
    # representation of view 1 and the j-th representation of view 2 for
    # scenes (i, j) in the batch.  The diagonal contains the similarities
    # of matching scenes, which explains our use of `diag_part` below to
    # get the normalized logits for matching scenes.
    similarities = tf.linalg.matmul(
        flat_1, flat_2, transpose_b=True, name="similarities"
    )

    # Rescale similarities to apply softmax temperature
    similarities = tf.divide(
        similarities, FLAGS.softmax_temperature, name="sharpened_similarities"
    )

    # NOTE: we use `reduce_sum` here and divide by the known batch size
    #       because each replica gets a smaller effective batch size in
    #       distributed training.
    with tf.name_scope("forward"):  # Predict view 2 from view 1
        softmax = tf.nn.log_softmax(similarities, axis=1, name="log_probabilities")
        nce_loss_forward = tf.negative(
            tf.reduce_sum(tf.linalg.diag_part(softmax))
            / csf.distribution.global_batch_size(),
            name="nce_loss_forward",
        )

    with tf.name_scope("backward"):  # Predict view 1 from view 2
        softmax = tf.nn.log_softmax(similarities, axis=0, name="log_probabilities")
        nce_loss_backward = tf.negative(
            tf.reduce_sum(tf.linalg.diag_part(softmax))
            / csf.distribution.global_batch_size(),
            name="nce_loss_backward",
        )

    return nce_loss_forward + nce_loss_backward


# TODO(Aidan): re-introduce: metrics, summaries, schedules, checkpoints
def run_unsupervised_training():
    """
    Perform a full unsupervised training run programmatically.
    """
    logging.info(
        "Starting unsupervised training run with flags:\n{}".format(
            FLAGS.flags_into_string()
        )
    )
    csf.distribution.initialize()
    data_shape = csf.data.data_shape()
    layers_and_weights = layer_loss_weights().items()

    logging.debug("Building dataset.")
    with csf.distribution.tpu_worker_context():
        dataset = csf.distribution.distribute_dataset_fn(csf.data.load_dataset)
        dataset_iterator = iter(dataset)

    with csf.distribution.tpu_worker_context(), csf.distribution.distributed_context():
        tf.random.set_seed(FLAGS.random_seed)

        logging.debug("Building model and optimizer.")
        encoder = resnet_encoder(gf.n_bands())
        encoder_vars = encoder.trainable_variables

        # TODO(Aidan): scheduled learning rate
        optimizer = tf.optimizers.Adam(1e-3, clipnorm=FLAGS.gradient_clipnorm)

        @csf.distribution.distribute_computation
        def _replicated_training_step(batch):
            batch = tf.reshape(batch, data_shape)
            with tf.name_scope("training_step"):
                with tf.name_scope("view_1"):
                    view_1 = _create_view(batch, seed=1)
                with tf.name_scope("view_2"):
                    view_2 = _create_view(batch, seed=2)

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

                gradients = tape.gradient(loss_total, encoder_vars)
                optimizer.apply_gradients(zip(gradients, encoder_vars))

        @tf.function
        def train_steps(iter_, steps):
            for _ in tf.range(steps):
                _replicated_training_step((next(iter_),))

        logging.info("Beginning unsupervised training.")
        while True:
            logging.info("Starting a step")
            train_steps(
                dataset_iterator,
                tf.convert_to_tensor(FLAGS.callback_frequency, dtype=tf.int32),
            )

        logging.info("Done with unsupervised training.")
