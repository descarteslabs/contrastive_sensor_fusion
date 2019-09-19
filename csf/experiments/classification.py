from absl import flags
import os.path
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Conv2D, Dense, Input, Lambda, GlobalMaxPooling2D, Flatten, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.layers import (Concatenate, Conv2D, Dense,
                                     GlobalMaxPooling2D)

from csf import global_flags as gf
from csf.experiments.data import N_OSM_LABELS, N_OSM_SAMPLES, load_osm_dataset
from csf.experiments.utils import encoder_head, LRMultiplierAdam, default_bands

FLAGS = flags.FLAGS

# Required parameters for all experiments
flags.DEFINE_string(
    "checkpoint_file",
    None,
    "Path to directory to load unsupervised weights from and save results to.",
)
flags.DEFINE_string(
    "osm_data_prefix", None, "Glob matching the prefix of OSM data to use."
)
flags.DEFINE_integer(
    "dataset_size", N_OSM_SAMPLES, "Size of dataset to use"
)

flags.mark_flags_as_required(["checkpoint_file", "osm_data_prefix"])

# Optional parameters with sensible defaults
flags.DEFINE_float("learning_rate", 1e-5, "Classification experiments' learning rate.")
flags.DEFINE_integer("batch_size", 8, "Batch size for classification experiments.")
flags.DEFINE_float("train_fraction", 0.8, "Fraction of OSM data used for training.")
flags.DEFINE_float("test_fraction", 0.1, "Fraction of OSM data used for testing.")
flags.DEFINE_float("val_fraction", 0.1, "Fraction of OSM data used for validation.")


def classification_model(size, n_labels, bands=None, batchsize=8, checkpoint_file=None):
    """Create a model based on the trained encoder (see encoder.py)
    for classification. Can operate on any subset of products or bands. """
    model_inputs, _, encoded = encoder_head(
        size,
        bands=bands,
        batchsize=batchsize,
        checkpoint_file=checkpoint_file,
        trainable=True
    )

    stack3 = encoded["conv4_block5_out"]
    stack4 = encoded["conv5_block3_out"]

    conv3 = Conv2D(filters=64, kernel_size=1)(stack3)
    conv4 = Conv2D(filters=128, kernel_size=1)(stack4)

    pooled3 = GlobalMaxPooling2D()(conv3)
    pooled4 = GlobalMaxPooling2D()(conv4)
    cat = Concatenate(axis=-1)([pooled3, pooled4])
    out = Dense(units=n_labels, activation='softmax', name='dense1')(cat)

    return Model(inputs=model_inputs, outputs=[out])


def classification_experiment():
    # Drop bands starting from high resolution to lower resolution
    band_indices = list()
    for band in FLAGS.bands:
        band_indices.append(default_bands.index(band))
    dataset_size = FLAGS.dataset_size or N_OSM_SAMPLES
    output_weights_prefix = 'classify_bands=%s_samples=%04i_weights=%s' % (
        ','.join(map(str, band_indices)),
        dataset_size,
        os.path.basename(FLAGS.checkpoint_file)
    )

    n_test_samples = int(N_OSM_SAMPLES * FLAGS.test_fraction)
    n_val_samples = int(N_OSM_SAMPLES * FLAGS.val_fraction)
    n_train_samples = min(dataset_size, N_OSM_SAMPLES - n_test_samples - n_val_samples)

    dataset = load_osm_dataset(FLAGS.osm_data_prefix, band_indices).shuffle(buffer_size=N_OSM_SAMPLES, seed=0)
    test_dataset = dataset.skip.take(n_test_samples)
    val_dataset = dataset.skip(n_test_samples).take(n_val_samples)
    train_dataset = dataset.skip(n_test_samples + n_val_samples).take(n_train_samples)

    train_dataset = (
        train_dataset.shuffle(buffer_size=n_train_samples)
        .batch(FLAGS.batch_size, drop_remainder=True)
        .repeat()
    )
    test_dataset = test_dataset.batch(FLAGS.batch_size, drop_remainder=True).repeat()
    val_dataset = val_dataset.batch(FLAGS.batch_size, drop_remainder=True).repeat()

    model = classification_model(
        size=128,
        n_labels=N_OSM_LABELS,
        bands=FLAGS.bands,
        batchsize=FLAGS.batch_size,
        checkpoint_file=FLAGS.checkpoint_file,
    )

    model.compile(
        optimizer=LRMultiplierAdam(
            learning_rate=FLAGS.learning_rate,
            clipnorm=1.0,
            multipliers={"dense1": 10.0}
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
                output_weights_prefix +
                '_top1={val_categorical_accuracy:.4f}_'
                'top2={val_top_k_categorical_accuracy:.4f}_'
                'epoch={epoch:02d}.h5',
                verbose=1,
                monitor='val_categorical_accuracy',
                mode='max',
                save_weights_only=True
            )
        ],
    )

    print("EVAL:")
    model.evaluate(test_dataset, steps=n_test_samples // FLAGS.batch_size)
