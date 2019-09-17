import glob
import os.path
import subprocess as sp
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Conv2D, Dense, Input, Lambda, GlobalMaxPooling2D
from tensorflow.keras import Model
import tensorflow.keras.backend as K

from csf.experiments.utils import default_bands, encoder_head


def classification_model(size, n_labels, bands=None, batchsize=8, checkpoint_file=None):
    """Create a model based on the trained encoder (see encoder.py)
    for classification. Can operate on any subset of products or bands. """
    model_inputs, _, encoded = encoder_head(
        size,
        bands=bands,
        batchsize=batchsize,
        checkpoint_file=checkpoint_file
    )

    stack3 = encoded['conv4_block5_out']
    stack4 = encoded['conv5_block3_out']

    conv3 = Conv2D(filters=64, kernel_size=1)(stack3)
    conv4 = Conv2D(filters=128, kernel_size=1)(stack4)

    pooled3 = GlobalMaxPooling2D()(conv3)
    pooled4 = GlobalMaxPooling2D()(conv4)

    cat = Concatenate(axis=-1)([pooled3, pooled4])
    dense = Dense(units=1000, activation='relu')(cat)
    out = Dense(units=n_labels, activation='sigmoid')(dense)

    return Model(inputs=model_inputs, outputs=[out])


def get_dataset(remote_prefix, n_labels, band_indices):
    features = {
        'spot_naip_phr': tf.io.FixedLenSequenceFeature([], dtype=tf.string, allow_missing=True),
        'label': tf.io.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True)
    }

    input_shape = (128, 128, 12)
    target_shape = (n_labels,)

    def _parse_image_function(example_proto):
        example_features = tf.io.parse_single_example(example_proto, features)
        image = tf.reshape(tf.io.decode_raw(example_features['spot_naip_phr'], tf.uint8), input_shape)
        bands_to_keep = list()
        for index in band_indices:
            bands_to_keep.append(tf.expand_dims(image[...,index], axis=-1))
        image = tf.concat(bands_to_keep, axis=-1)
        target = tf.reshape(tf.one_hot(example_features['label'], depth=n_labels), target_shape)
        return image, target

    if remote_prefix.startswith('***REMOVED***'):
        tfrecord_paths = sp.check_output(('gsutil', '-m', 'ls', remote_prefix)).decode("ascii").split('\n')
    else:
        tfrecord_paths = [filename for filename in glob.glob(remote_prefix) if os.path.isfile(filename)]
    dataset = tf.data.TFRecordDataset(tfrecord_paths)
    return dataset.map(_parse_image_function)


def degrading_inputs_experiment():
    batchsize = 8
    n_labels = 12
    n_total_samples = 8600

    # Drop bands starting from high resolution to lower resolution
    for n_bands in range(12, 0, -1):
        band_indices = list(range(n_bands))

        # Provides 4-band SPOT, NAIP, PHR images and OSM labels:
        #dataset = get_dataset('***REMOVED***', n_labels=n_labels, n_bands=n_bands)
        # Streaming from google storage is bugging out, so we download locally first:
        dataset = get_dataset('./osm_data/osm_*.tfrecord', n_labels=n_labels, band_indices=band_indices)

        n_train_samples = int(n_total_samples * 0.8)
        n_test_samples = int(n_total_samples * 0.1)
        n_val_samples = int(n_total_samples * 0.1)

        train_dataset = dataset.take(n_train_samples)
        test_dataset = dataset.take(n_test_samples)
        val_dataset = dataset.take(n_val_samples)

        train_dataset = dataset.shuffle(buffer_size=n_train_samples).batch(batchsize).repeat()
        test_dataset = test_dataset.batch(batchsize).repeat()
        val_dataset = val_dataset.batch(batchsize).repeat()

        checkpoint_file = '***REMOVED***outputs/basenets_fusion_tpu_deploy_1/ckpt-80'
        model = classification_model(
            size=128,
            n_labels=n_labels,
            bands=default_bands[:n_bands],
            batchsize=batchsize,
            checkpoint_file=checkpoint_file
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[tf.keras.metrics.CategoricalAccuracy(),
                     tf.keras.metrics.TopKCategoricalAccuracy(k=2),
            ]
        )

        model.fit(
            train_dataset,
            epochs=64,
            steps_per_epoch=n_train_samples // batchsize,
            validation_data=val_dataset,
            validation_steps=n_val_samples // batchsize,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    'classification-%02dband-{epoch:02d}-{val_categorical_accuracy:.4f}.h5' % (n_bands,),
                    verbose=1,
                    mode='max',
                    save_weights_only=True
                )
            ]
        )
        model.evaluate(test_dataset, steps=n_test_samples // batchsize)


def degrading_dataset_experiment():
    batchsize = 8
    n_labels = 12
    n_total_samples = 8600
    n_bands = 12

    # Drop dataset samples
    for n_samples_keep in (8000, 6000, 4000, 2000, 1000, 500, 250):
        band_indices = list(range(n_bands))

        # Provides 4-band SPOT, NAIP, PHR images and OSM labels:
        #dataset = get_dataset('***REMOVED***', n_labels=n_labels, n_bands=n_bands)
        # Streaming from google storage is bugging out, so we download locally first:
        dataset = get_dataset('./osm_data/osm_*.tfrecord', n_labels=n_labels, band_indices=band_indices)

        n_train_samples = int(n_samples_keep * 0.8)
        n_test_samples = int(n_samples_keep * 0.1)
        n_val_samples = int(n_samples_keep * 0.1)

        train_dataset = dataset.take(n_train_samples)
        test_dataset = dataset.skip(n_train_samples).take(n_test_samples)
        val_dataset = dataset.skip(n_train_samples + n_test_samples).take(n_val_samples)

        train_dataset = dataset.shuffle(buffer_size=n_train_samples).batch(batchsize).repeat()
        test_dataset = test_dataset.batch(batchsize).repeat()
        val_dataset = val_dataset.batch(batchsize).repeat()

        checkpoint_file = '***REMOVED***outputs/basenets_fusion_tpu_deploy_1/ckpt-80'
        model = classification_model(
            size=128,
            n_labels=n_labels,
            bands=default_bands[:n_bands],
            batchsize=batchsize,
            checkpoint_file=checkpoint_file
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[tf.keras.metrics.CategoricalAccuracy(),
                     tf.keras.metrics.TopKCategoricalAccuracy(k=2),
            ]
        )

        model.fit(
            train_dataset,
            epochs=64,
            steps_per_epoch=n_train_samples // batchsize,
            validation_data=val_dataset,
            validation_steps=n_val_samples // batchsize,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    'classification_%04dsamples_{val_categorical_accuracy:.4f}_'
                    '{val_top_k_categorical_accuracy:.4f}_epoch{epoch:02d}.h5' % (n_samples_keep,),
                    verbose=1,
                    mode='max',
                    save_weights_only=True,
                    save_best_only=True
                )
            ]
        )
        model.evaluate(test_dataset, steps=n_test_samples // batchsize)


if __name__ == '__main__':
    #degrading_inputs_experiment()
    degrading_dataset_experiment()
