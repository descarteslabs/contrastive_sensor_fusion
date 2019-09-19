import tensorflow as tf
from absl import flags
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten

from csf import global_flags as gf  # noqa
from csf.experiments.data import N_OSM_LABELS, load_osm_dataset
from csf.experiments.utils import default_bands, encoder_head

FLAGS = flags.FLAGS

# Required parameters for all experiments
flags.DEFINE_string(
    "checkpoint_file",
    None,
    "Path to directory to load unsupervised weights from and save results to.",
)
flags.DEFINE_list("experiment_bands", None, "Bands used for creating representations.")
flags.DEFINE_string(
    "osm_data_train", None, "Glob matching the path of OSM training data to use."
)
flags.DEFINE_string(
    "osm_data_val", None, "Glob matching the path of OSM validation data to use."
)

flags.mark_flags_as_required(
    ["checkpoint_file", "osm_data_train", "osm_data_val", "experiment_bands"]
)

# Optional parameters with sensible defaults
flags.DEFINE_float("learning_rate", 1e-5, "Classification experiments' learning rate.")
flags.DEFINE_integer("batch_size", 32, "Batch size for classification experiments.")


def classification_model(size, n_labels, bands=None, batchsize=8, checkpoint_file=None):
    """Create a model based on the trained encoder (see encoder.py)
    for classification. Can operate on any subset of products or bands. """
    model_inputs, _, encoded = encoder_head(
        size,
        bands=bands,
        batchsize=batchsize,
        checkpoint_file=checkpoint_file,
        trainable=False,
    )

    pooled = Flatten()(encoded["conv4_block5_out"])
    out = Dense(units=n_labels, activation="softmax", name="linear_out")(pooled)

    return Model(inputs=model_inputs, outputs=[out])


def classification_experiment():
    # Drop bands starting from high resolution to lower resolution
    band_indices = list()
    for band in FLAGS.experiment_bands:
        band_indices.append(default_bands.index(band))

    train_dataset = (
        load_osm_dataset(FLAGS.osm_data_train, band_indices)
        .shuffle(buffer_size=1024)
        .batch(FLAGS.batch_size, drop_remainder=True)
    )

    val_dataset = load_osm_dataset(FLAGS.osm_data_val, band_indices).batch(
        FLAGS.batch_size, drop_remainder=True
    )

    model = classification_model(
        size=128,
        n_labels=N_OSM_LABELS,
        bands=FLAGS.experiment_bands,
        batchsize=FLAGS.batch_size,
        checkpoint_file=FLAGS.checkpoint_file,
    )

    optimizer = tf.optimizers.Adam(learning_rate=FLAGS.learning_rate, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(),
            tf.keras.metrics.TopKCategoricalAccuracy(k=2),
        ],
    )

    model.fit(train_dataset, epochs=1)
    model.evaluate(val_dataset)
