import tensorflow as tf
from absl import flags
from tensorflow.keras import Model
from tensorflow.keras.layers import (Concatenate, Conv2D, Dense,
                                     GlobalMaxPooling2D)

from csf import global_flags as gf
from csf.experiments.data import N_OSM_LABELS, N_OSM_SAMPLES, load_osm_dataset
from csf.experiments.utils import default_bands, encoder_head

FLAGS = flags.FLAGS

# Required parameters for all experiments
flags.DEFINE_string(
    "checkpoint_dir",
    None,
    "Path to directory to load unsupervised weights from and save results to.",
)
flags.DEFINE_string(
    "osm_data_prefix", None, "Glob matching the prefix of OSM data to use."
)

flags.mark_flags_as_required(["batch_size", "checkpoint_dir", "osm_data_prefix"])

# Optional parameters with sensible defaults
flags.DEFINE_float("learning_rate", 1e-5, "Classification experiments' learning rate.")
flags.DEFINE_integer("batch_size", 8, "Batch size for classification experiments.")
flags.DEFINE_float("train_fraction", 0.8, "Fraction of OSM data used for training.")
flags.DEFINE_float("test_fraction", 0.1, "Fraction of OSM data used for testing.")
flags.DEFINE_float("val_fraction", 0.1, "Fraction of OSM data used for validation.")


def classification_model(size, n_labels, bands=None, batchsize=8, checkpoint_dir=None):
    """Create a model based on the trained encoder (see encoder.py)
    for classification. Can operate on any subset of products or bands. """
    model_inputs, _, encoded = encoder_head(
        size, bands=bands, batchsize=batchsize, checkpoint_dir=checkpoint_dir
    )

    stack3 = encoded["conv4_block5_out"]
    stack4 = encoded["conv5_block3_out"]

    conv3 = Conv2D(filters=64, kernel_size=1)(stack3)
    conv4 = Conv2D(filters=128, kernel_size=1)(stack4)

    pooled3 = GlobalMaxPooling2D()(conv3)
    pooled4 = GlobalMaxPooling2D()(conv4)

    cat = Concatenate(axis=-1)([pooled3, pooled4])
    dense = Dense(units=1000, activation="relu")(cat)
    out = Dense(units=n_labels, activation="sigmoid")(dense)

    return Model(inputs=model_inputs, outputs=[out])


def degrading_inputs_experiment():
    # Drop bands starting from high resolution to lower resolution
    for n_bands in range(gf.n_bands(), 0, -1):
        band_indices = list(range(n_bands))

        n_train_samples = int(N_OSM_SAMPLES * FLAGS.train_fraction)
        n_test_samples = int(N_OSM_SAMPLES * FLAGS.test_fraction)
        n_val_samples = int(N_OSM_SAMPLES * FLAGS.val_fraction)

        dataset = load_osm_dataset(FLAGS.osm_data_prefix, band_indices)
        train_dataset = dataset.take(n_train_samples)
        test_dataset = dataset.take(n_test_samples)
        val_dataset = dataset.take(n_val_samples)

        train_dataset = (
            dataset.shuffle(buffer_size=n_train_samples)
            .batch(FLAGS.batch_size)
            .repeat()
        )
        test_dataset = test_dataset.batch(FLAGS.batch_size).repeat()
        val_dataset = val_dataset.batch(FLAGS.batch_size).repeat()

        model = classification_model(
            size=128,
            n_labels=N_OSM_LABELS,
            bands=default_bands[:n_bands],
            batchsize=FLAGS.batch_size,
            checkpoint_dir=FLAGS.checkpoint_dir,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=FLAGS.learning_rate, clipnorm=1.0
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(),
                tf.keras.metrics.TopKCategoricalAccuracy(k=2),
            ],
        )

        model.fit(
            train_dataset,
            epochs=64,
            steps_per_epoch=n_train_samples // FLAGS.batch_size,
            validation_data=val_dataset,
            validation_steps=n_val_samples // FLAGS.batch_size,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    "classification-%02dband-{epoch:02d}"
                    "-{val_categorical_accuracy:.4f}.h5" % (n_bands,),
                    verbose=1,
                    mode="max",
                    save_weights_only=True,
                )
            ],
        )
        model.evaluate(test_dataset, steps=n_test_samples // FLAGS.batch_size)


def degrading_dataset_experiment():
    # Drop dataset samples
    for n_samples_keep in (N_OSM_SAMPLES // 3, 2 * N_OSM_SAMPLES // 3, N_OSM_SAMPLES):
        band_indices = list(range(gf.n_bands()))

        n_train_samples = int(n_samples_keep * FLAGS.train_fraction)
        n_test_samples = int(n_samples_keep * FLAGS.test_fraction)
        n_val_samples = int(n_samples_keep * FLAGS.val_fraction)

        dataset = load_osm_dataset(FLAGS.osm_data_prefix, band_indices=band_indices)
        train_dataset = dataset.take(n_train_samples)
        test_dataset = dataset.take(n_test_samples)
        val_dataset = dataset.take(n_val_samples)

        train_dataset = (
            dataset.shuffle(buffer_size=n_train_samples)
            .batch(FLAGS.batch_size)
            .repeat()
        )
        test_dataset = test_dataset.batch(FLAGS.batch_size).repeat()
        val_dataset = val_dataset.batch(FLAGS.batch_size).repeat()

        model = classification_model(
            size=128,
            n_labels=N_OSM_LABELS,
            bands=default_bands[: gf.n_bands()],
            batchsize=FLAGS.batch_size,
            checkpoint_dir=FLAGS.checkpoint_dir,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=FLAGS.learning_rate, clipnorm=1.0
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(),
                tf.keras.metrics.TopKCategoricalAccuracy(k=2),
            ],
        )

        model.fit(
            train_dataset,
            epochs=64,
            steps_per_epoch=n_train_samples // FLAGS.batch_size,
            validation_data=val_dataset,
            validation_steps=n_val_samples // FLAGS.batch_size,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    "classification-%04dsample-{epoch:02d}"
                    "-{val_categorical_accuracy:.4f}.h5" % (n_samples_keep,),
                    verbose=1,
                    mode="max",
                    save_weights_only=True,
                )
            ],
        )
        model.evaluate(test_dataset, steps=n_test_samples // FLAGS.batch_size)
