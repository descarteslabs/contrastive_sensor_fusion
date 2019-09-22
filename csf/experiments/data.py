"""
Utilities for loading data relevant to experiment code.
"""

import tensorflow as tf

from csf import global_flags as gf

OSM_TILESIZE = 128
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


def load_osm_dataset(tfrecord_glob, band_indices):
    """
    Load the OpenStreetMap classification dataset.

    Parameters
    ----------
    tfrecord_glob : string
        A glob specifying the path of files used to load the dataset.
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
    target_shape = (len(OSM_CLASSES),)

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
            tf.one_hot(example_features["label"], depth=len(OSM_CLASSES)), target_shape
        )
        image = (tf.cast(image, tf.dtypes.float32) / 128.0) - 1.0
        return image, target

    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_vectorization.enabled = True

    return (
        tf.data.Dataset.list_files(tfrecord_glob)
        .interleave(tf.data.TFRecordDataset)
        .with_options(options)
        .map(_parse_image_function)
        .cache()
    )
