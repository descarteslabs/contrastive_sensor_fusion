"""
Utilities for loading data relevant to experiment code.
"""

import glob
import os.path
import subprocess as sp

import tensorflow as tf

from csf import global_flags as gf

OSM_TILESIZE = 128
N_OSM_LABELS = 12
N_OSM_SAMPLES = 8600

BUILDINGS_TILESIZE = 512
N_BUILDINGS_TRAIN_SAMPLES = 12000
N_BUILDINGS_VAL_SAMPLES = 12000
N_BUILDINGS_TEST_SAMPLES = 12000


def load_osm_dataset(remote_prefix, band_indices):
    """
    Load the OpenStreetMap classification dataset.

    Parameters
    ----------
    remote_prefix : string
        A glob specifying the prefix of files used to load the dataset.
    band_indices : [int]
        List of input bands to keep.

    Returns
    -------
    tf.data.Dataset
        The dataset, yielding (image, label) pairs.
    """
    features = {
        "spot_naip_phr": tf.io.FixedLenSequenceFeature(
            [], dtype=tf.string, allow_missing=True
        ),
        "label": tf.io.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
    }

    input_shape = (OSM_TILESIZE, OSM_TILESIZE, gf.n_bands())
    target_shape = (N_OSM_LABELS,)

    def _parse_image_function(example_proto):
        example_features = tf.io.parse_single_example(example_proto, features)
        image = tf.reshape(
            tf.io.decode_raw(example_features["spot_naip_phr"], tf.uint8), input_shape
        )
        bands_to_keep = list()
        for index in band_indices:
            bands_to_keep.append(tf.expand_dims(image[..., index], axis=-1))
        image = tf.concat(bands_to_keep, axis=-1)
        target = tf.reshape(
            tf.one_hot(example_features["label"], depth=N_OSM_LABELS), target_shape
        )
        image = (tf.cast(image, tf.dtypes.float32) / 128.0) - 1.0
        return image, target

    if remote_prefix.startswith("***REMOVED***"):
        tfrecord_paths = (
            sp.check_output(("gsutil", "-m", "ls", remote_prefix))
            .decode("ascii")
            .split("\n")
        )
    else:
        tfrecord_paths = [
            filename
            for filename in glob.glob(remote_prefix)
            if os.path.isfile(filename)
        ]
    dataset = tf.data.TFRecordDataset(tfrecord_paths)
    return dataset.map(_parse_image_function)


def load_buildings_dataset(remote_prefix, target_tilesize, experiment_bands):
    """
    Load the buildings segmentation dataset.

    Parameters
    ----------
    remote_prefix : string
        A glob specifying the prefix of files used to load the dataset.
    target_tilesize : int
        Size of tiles to yield from the dataset.
    experiment_bands : [string]
        Names of bands to use for input, in the order they should be produced.

    Returns
    -------
    tf.data.Dataset
        The dataset, yielding (image, segmentation mask) pairs.
    """
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

    input_shape = (BUILDINGS_TILESIZE, BUILDINGS_TILESIZE, len(experiment_bands))
    target_shape = (BUILDINGS_TILESIZE, BUILDINGS_TILESIZE, 1)

    # We need to upsample NAIP to the target resolution of 0.5m from 1.0m
    upsample_size = 2 * BUILDINGS_TILESIZE

    def _parse_image_function(example_proto):
        example_features = tf.io.parse_single_example(example_proto, features)
        image = tf.reshape(example_features["image/image_data"], input_shape)
        target = tf.reshape(example_features["target/target_data"], target_shape)
        image = (tf.cast(image, tf.dtypes.float32) / 128.0) - 1.0
        image = tf.image.resize(
            image, size=(upsample_size, upsample_size), method="bilinear"
        )
        target = tf.image.resize(
            target, size=(upsample_size, upsample_size), method="bilinear"
        )
        images = list()
        targets = list()
        for j in range(upsample_size // target_tilesize):
            for i in range(upsample_size // target_tilesize):
                images.append(
                    image[
                        j * target_tilesize : (j + 1) * target_tilesize,
                        i * target_tilesize : (i + 1) * target_tilesize,
                        :,
                    ]
                )
                targets.append(
                    target[
                        j * target_tilesize : (j + 1) * target_tilesize,
                        i * target_tilesize : (i + 1) * target_tilesize,
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
    return dataset.map(
        _parse_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
