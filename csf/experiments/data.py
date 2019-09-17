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
OSM_CLASSES = [
    "Bridge",
    "Breakwater",
    "Farm",
    "Substation",
    "Stadium",
    "Golf course",
    "Dam",
    "Quarry",
    "Farmland",
    "Forest",
    "Water",
    "Bare rock",
]

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

    if remote_prefix.startswith("gs://"):
        tfrecord_paths = (
            sp.check_output(("gsutil", "-m", "ls", remote_prefix))
            .decode("ascii")
            .splitlines()
        )
    else:
        tfrecord_paths = [
            filename
            for filename in glob.glob(remote_prefix)
            if os.path.isfile(filename)
        ]
    dataset = tf.data.TFRecordDataset(tfrecord_paths)
    return dataset.map(_parse_image_function)
