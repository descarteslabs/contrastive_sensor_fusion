"""
Utilities to load data for the unsupervised learning task.
"""

import csf.global_flags as gf
import tensorflow as tf
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "unsupervised_batch_size",
    None,
    "Batch size to use for unsupervised training.",
    lower_bound=2,
)
flags.DEFINE_integer(
    "unsupervised_data_tilesize",
    None,
    "Tilesize of data used for unsupervised learning.",
    lower_bound=1,
)
flags.DEFINE_string(
    "unsupervised_data_feature_name",
    None,
    "Name of the key in the key-value dictionary containing unsupervised examples.",
)
flags.mark_flags_as_required(
    [
        "unsupervised_batch_size",
        "unsupervised_data_tilesize",
        "unsupervised_data_feature_name",
    ]
)


# Optional flags to configure dataset loading
flags.DEFINE_integer(
    "shuffle_buffer_size", 8000, "Number of examples to hold in the shuffle buffer."
)
flags.DEFINE_integer(
    "tfrecord_buffer_size", 8388608, "Bytes of input buffering for each tfrecord file."
)
flags.DEFINE_integer(
    "tfrecord_parallel_reads", 64, "Number of tfrecord files to read in parallel."
)


def load_dataset(data_file=None, data_listing=None):
    """
    Get the dataset used for unsupervised learning.

    The dataset consists of examples, which are tensors of input imagery concatenated,
    serialized together into a byte string, and then saved as tf Examples with a single
    key (see: unsupervised_data_feature_name).

    Parameters
    ----------
    data_file : None, str, or [str], optional
        If str, path to a tfrecord to use as the unsupervised data.
        If [str], a list of such paths.
        If None, `data_listing` must be specified and is used instead.
    data_listing : None or str, optional
        If str, path to a newline-separated file specifying tfrecord paths.
        If None, `data_file` must be specified and is used instead.

    Returns
    -------
    tf.data.Dataset
        A dataset of batched examples, where each example is a set of coterminous
        input bands.
    """
    if (data_file is None and data_listing is None) or (
        data_file is not None and data_listing is not None
    ):
        raise ValueError(
            "Exactly one of `data_file` or `data_listing` must be specified"
        )

    feature_spec = {
        FLAGS.unsupervised_data_feature_name: tf.io.FixedLenFeature(
            [], tf.dtypes.string
        )
    }

    batch_shape = (
        FLAGS.unsupervised_batch_size,
        FLAGS.unsupervised_data_tilesize,
        FLAGS.unsupervised_data_tilesize,
        gf.n_bands(),
    )

    def load_tfrecord(filenames):
        return tf.data.TFRecordDataset(
            filenames,
            buffer_size=FLAGS.tfrecord_buffer_size,
            num_parallel_reads=FLAGS.tfrecord_parallel_reads,
        )

    def preprocess_batch(batch):
        batch = tf.io.parse_example(batch, feature_spec)[
            FLAGS.unsupervised_data_feature_name
        ]
        batch = tf.io.decode_raw(batch, tf.dtypes.uint8)
        batch = tf.reshape(batch, batch_shape)

        # Apply augmentation with flips and rotations
        which_aug = tf.random.uniform(
            shape=(), dtype=tf.dtypes.int32, minval=0, maxval=8
        )
        aug_options = {
            0: lambda: batch,
            1: lambda: tf.transpose(batch, perm=[0, 2, 1, 3]),
            2: lambda: tf.image.flip_up_down(batch),
            3: lambda: tf.image.flip_left_right(tf.image.rot90(batch, k=1)),
            4: lambda: tf.image.flip_left_right(batch),
            5: lambda: tf.image.rot90(batch, k=1),
            6: lambda: tf.image.rot90(batch, k=2),
            7: lambda: tf.image.rot90(batch, k=3),
        }
        batch = tf.switch_case(which_aug, aug_options)

        batch = (tf.cast(batch, tf.float32) - 128.0) / 128.0
        return batch

    if data_file:  # Dataset specified as a single tfrecord or small list of tfrecords
        ds = load_tfrecord(data_file).repeat()
    else:  # Dataset specified as a file listing tfrecords
        ds = (
            tf.data.experimental.CsvDataset(data_listing, [tf.dtypes.string])
            .repeat()
            .interleave(
                load_tfrecord,
                cycle_length=FLAGS.tfrecord_parallel_reads,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        )

    ds = (
        ds.shuffle(FLAGS.shuffle_buffer_size, reshuffle_each_iteration=True)
        .batch(FLAGS.unsupervised_batch_size)
        .map(preprocess_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    return ds
