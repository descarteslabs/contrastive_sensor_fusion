import tensorflow as tf
import tensorflow.keras.backend as K
from absl import flags
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Conv2D, UpSampling2D

from csf.experiments.data import (N_BUILDINGS_TEST_SAMPLES,
                                  N_BUILDINGS_TRAIN_SAMPLES,
                                  N_BUILDINGS_VAL_SAMPLES,
                                  load_buildings_dataset)
from csf.experiments.utils import encoder_head

FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_integer("epochs", None, "Numer of epochs to train experiment for.")
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

# Optional parameters, with sensible defaults.
flags.DEFINE_float("learning_rate", 1e-4, "Hypercolumn experiments' learning rate.")
flags.DEFINE_integer("batch_size", 16, "Batch size for hypercolumn experiments.")

flags.mark_flags_as_required(
    [
        "epochs",
        "target_tilesize",
        "checkpoint_dir",
        "experiment_bands",
        "train_dataset",
        "val_dataset",
        "test_dataset",
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
    train_dataset = load_buildings_dataset(
        FLAGS.train_dataset, N_BUILDINGS_TRAIN_SAMPLES, FLAGS.experiment_bands
    )
    val_dataset = load_buildings_dataset(
        FLAGS.val_dataset, N_BUILDINGS_VAL_SAMPLES, FLAGS.experiment_bands
    )
    test_dataset = load_buildings_dataset(
        FLAGS.test_dataset, N_BUILDINGS_TEST_SAMPLES, FLAGS.experiment_bands
    )

    train_dataset = train_dataset.shuffle(buffer_size=750).batch(FLAGS.batch_size)
    val_dataset = val_dataset.batch(FLAGS.batch_size)
    test_dataset = test_dataset.batch(FLAGS.batch_size)

    model.fit(
        train_dataset,
        epochs=FLAGS.epochs,
        steps_per_epoch=N_BUILDINGS_TRAIN_SAMPLES // FLAGS.batch_size,
        validation_data=val_dataset,
        validation_steps=N_BUILDINGS_VAL_SAMPLES // FLAGS.batch_size,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                "hypercolumns-{epoch:02d}-{val_f1_metric:.4f}.h5",
                verbose=1,
                mode="max",
                save_weights_only=True,
            )
        ],
    )
    model.evaluate(test_dataset, N_BUILDINGS_TEST_SAMPLES // FLAGS.batch_size)


def imagenet_comparison_experiment():
    pass  # TODO
