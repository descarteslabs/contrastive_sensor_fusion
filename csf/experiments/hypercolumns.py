import subprocess as sp

import tensorflow as tf
import tensorflow.keras.backend as K
from absl import flags
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Conv2D, UpSampling2D

from csf.experiments.utils import encoder_head

FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_integer("epochs", None, "Numer of epochs to train experiment for.")
flags.DEFINE_integer("initial_tilesize", None, "Size of the raw data tiles.")
flags.DEFINE_integer(
    "target_tilesize",
    None,
    "Size of the tiles the model will use. Tiles bigger than this are cropped.",
)
flags.DEFINE_string(
    "checkpoint_dir",
    None,
    "Path to directory to load unsupervised weights from and save results to.",
)
flags.DEFINE_list("experiment_bands", None, "Bands used for downstream experiments.")

flags.DEFINE_string("train_dataset", None, "Glob to tfrecords used for training.")
flags.DEFINE_string("val_dataset", None, "Glob to tfrecords used for validation.")
flags.DEFINE_string("test_dataset", None, "Glob to tfrecords used for testing.")

flags.DEFINE_integer("n_train_samples", None, "Number of samples used for training.")
flags.DEFINE_integer("n_val_samples", None, "Number of samples used for validation.")
flags.DEFINE_integer("n_test_samples", None, "Number of samples used for testing.")

# Optional parameters, with sensible defaults.
flags.DEFINE_float("learning_rate", 1e-4, "Hypercolumn experiments' learning rate.")
flags.DEFINE_integer("batch_size", 16, "Batch size for hypercolumn experiments.")

flags.mark_flags_as_required(
    [
        "epochs",
        "initial_tilesize",
        "target_tilesize",
        "checkpoint_dir",
        "experiment_bands",
        "train_dataset",
        "val_dataset",
        "test_dataset",
        "n_train_samples",
        "n_val_samples",
        "n_test_samples",
    ]
)


def hypercolumn_model(size, bands=None, batchsize=8, checkpoint_dir=None):
    """Create a model based on the trained encoder (see encoder.py)
    for semantic segmentation. Can operate on any subset of bands. """
    model_inputs, scaled_inputs, encoded = encoder_head(
        size, bands=bands, batchsize=batchsize, checkpoint_dir=checkpoint_dir
    )

    stack1 = encoded["conv2_block2_out"]
    stack2 = encoded["conv3_block3_out"]
    stack3 = encoded["conv4_block5_out"]
    stack4 = encoded["conv5_block3_out"]

    up0 = scaled_inputs
    up1 = UpSampling2D(size=(128 // 32), interpolation="bilinear")(stack1)
    up2 = UpSampling2D(size=(128 // 16), interpolation="bilinear")(stack2)
    up3 = UpSampling2D(size=(128 // 8), interpolation="bilinear")(stack3)
    up4 = UpSampling2D(size=(128 // 4), interpolation="bilinear")(stack4)

    cat = Concatenate(axis=-1)([up0, up1, up2, up3, up4])
    conv = Conv2D(filters=1000, kernel_size=1, activation="relu")(cat)
    out = Conv2D(filters=1, kernel_size=1, activation="sigmoid")(conv)

    return Model(inputs=model_inputs, outputs=[out])


def edge_weighted_binary_crossentropy(
    y_true, y_pred, edge_weight=2.0, interior_weight=1.0, exterior_weight=0.5
):
    dilation = tf.nn.max_pool2d(y_true, ksize=3, strides=1, padding="SAME")
    interior = 1.0 - tf.nn.max_pool2d(1.0 - y_true, ksize=3, strides=1, padding="SAME")
    edge = dilation - interior
    exterior = 1.0 - dilation

    true_at_edge = y_true * edge
    true_at_interior = y_true * interior
    true_at_exterior = y_true * exterior

    edge_loss = tf.keras.losses.binary_crossentropy(true_at_edge, y_pred * edge)
    interior_loss = tf.keras.losses.binary_crossentropy(
        true_at_interior, y_pred * interior
    )
    exterior_loss = tf.keras.losses.binary_crossentropy(
        true_at_exterior, y_pred * exterior
    )

    return (
        edge_weight * edge_loss
        + interior_weight * interior_loss
        + exterior_weight * exterior_loss
    )


def recall_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_metric(y_true, y_pred):
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall + K.epsilon())


def get_dataset(remote_prefix):
    features = {
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/channels": tf.io.FixedLenFeature([], tf.int64),
        "image/colorspace": tf.io.FixedLenFeature([], tf.string),
        "image/format": tf.io.FixedLenFeature([], tf.string),
        "image/filename": tf.io.FixedLenFeature([], tf.string),
        "image/image_data": tf.io.FixedLenSequenceFeature(
            [], dtype=tf.float32, allow_missing=True
        ),
        "target/height": tf.io.FixedLenFeature([], tf.int64),
        "target/width": tf.io.FixedLenFeature([], tf.int64),
        "target/channels": tf.io.FixedLenFeature([], tf.int64),
        "target/target_data": tf.io.FixedLenSequenceFeature(
            [], dtype=tf.float32, allow_missing=True
        ),
    }

    input_shape = (
        FLAGS.initial_tilesize,
        FLAGS.initial_tilesize,
        len(FLAGS.experiment_bands),
    )
    target_shape = (FLAGS.initial_tilesize, FLAGS.initial_tilesize, 1)

    # We need to upsample NAIP to the target resolution of 0.5m from 1.0m
    upsample_size = 2 * FLAGS.initial_tilesize

    def _parse_image_function(example_proto):
        example_features = tf.io.parse_single_example(example_proto, features)
        image = tf.reshape(example_features["image/image_data"], input_shape)
        target = tf.reshape(example_features["target/target_data"], target_shape)
        image /= 128.0
        image -= 1.0
        image = tf.image.resize(
            image, size=(upsample_size, upsample_size), method="bilinear"
        )
        target = tf.image.resize(
            target, size=(upsample_size, upsample_size), method="bilinear"
        )
        images = list()
        targets = list()
        for j in range(upsample_size // FLAGS.target_tilesize):
            for i in range(upsample_size // FLAGS.target_tilesize):
                images.append(
                    image[
                        j * FLAGS.target_tilesize : (j + 1) * FLAGS.target_tilesize,
                        i * FLAGS.target_tilesize : (i + 1) * FLAGS.target_tilesize,
                        :,
                    ]
                )
                targets.append(
                    target[
                        j * FLAGS.target_tilesize : (j + 1) * FLAGS.target_tilesize,
                        i * FLAGS.target_tilesize : (i + 1) * FLAGS.target_tilesize,
                        :,
                    ]
                )
        images = tf.data.Dataset.from_tensors(images)
        targets = tf.data.Dataset.from_tensors(targets)
        return tf.data.Dataset.zip((images, targets))

    tfrecord_paths = (
        sp.check_output(("gsutil", "-m", "ls", remote_prefix))
        .decode("ascii")
        .split("\n")
    )
    dataset = tf.data.TFRecordDataset(tfrecord_paths)
    return dataset.interleave(
        _parse_image_function,
        cycle_length=tf.data.experimental.AUTOTUNE,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    ).unbatch()


def sensior_fusion_experiment():
    model = hypercolumn_model(
        size=128,
        bands=FLAGS.experiment_bands,
        batchsize=FLAGS.batch_size,
        checkpoint_dir=FLAGS.checkpoint_dir,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=FLAGS.learning_rate, clipnorm=1.0
        ),
        loss=edge_weighted_binary_crossentropy,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            f1_metric,
        ],
    )

    # Provides 4-band NAIP images and targets:
    train_dataset = get_dataset(FLAGS.train_dataset)
    val_dataset = get_dataset(FLAGS.val_dataset)
    test_dataset = get_dataset(FLAGS.test_dataset)

    train_dataset = train_dataset.shuffle(buffer_size=750).batch(FLAGS.batch_size)
    val_dataset = val_dataset.batch(FLAGS.batch_size)
    test_dataset = test_dataset.batch(FLAGS.batch_size)

    model.fit(
        train_dataset,
        epochs=FLAGS.epochs,
        steps_per_epoch=FLAGS.n_train_samples // FLAGS.batch_size,
        validation_data=val_dataset,
        validation_steps=FLAGS.n_val_samples // FLAGS.batch_size,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                "hypercolumns-{epoch:02d}-{val_f1_metric:.4f}.h5",
                verbose=1,
                mode="max",
                save_weights_only=True,
            )
        ],
    )
    model.evaluate(test_dataset, FLAGS.n_test_samples // FLAGS.batch_size)


def imagenet_comparison_experiment():
    pass  # TODO
