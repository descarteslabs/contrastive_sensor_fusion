"""
Utilities to load data for the unsupervised learning task.
"""

import tensorflow as tf
from absl import flags

import csf.global_flags as gf

FLAGS = flags.FLAGS

# Required arguments
flags.DEFINE_integer(
    "batch_size",
    None,
    "Batch size to use for unsupervised training. "
    "In the distributed context, this is the global batch size.",
    lower_bound=1,
)
flags.DEFINE_integer(
    "data_tilesize",
    None,
    "Tilesize of data used for unsupervised learning.",
    lower_bound=1,
)
flags.DEFINE_string(
    "data_feature_name",
    None,
    "Name of the key in the key-value dictionary containing unsupervised examples.",
)
flags.mark_flags_as_required(["batch_size", "data_tilesize", "data_feature_name"])

# Exactly one of these must be defined
flags.DEFINE_string(
    "data_file", None, "Path to a tfrecord to use as unsupervised data."
)
flags.DEFINE_string(
    "data_listing", None, "Path to a newline-separated file specifying tfrecord paths."
)
flags.mark_flags_as_mutual_exclusive(["data_file", "data_listing"], required=True)


# Optional flags to configure dataset loading
flags.DEFINE_integer(
    "shuffle_buffer_size",
    8000,
    "Number of examples to hold in the shuffle buffer. If <= 0, do not shuffle data.",
)
flags.DEFINE_integer(
    "tfrecord_buffer_size", 8388608, "Bytes of input buffering for each tfrecord file."
)
flags.DEFINE_integer(
    "tfrecord_parallel_reads", 64, "Number of tfrecord files to read in parallel."
)


def _load_tfrecord(filenames):
    return tf.data.TFRecordDataset(
        filenames,
        buffer_size=FLAGS.tfrecord_buffer_size,
        num_parallel_reads=FLAGS.tfrecord_parallel_reads,
    )


@tf.function
def _preprocess_batch(batch):
    feature_spec = {
        FLAGS.data_feature_name: tf.io.FixedLenFeature([], tf.dtypes.string)
    }

    batch_shape = (
        FLAGS.batch_size,
        FLAGS.data_tilesize,
        FLAGS.data_tilesize,
        gf.n_bands(),
    )

    batch = tf.io.parse_example(batch, feature_spec)[FLAGS.data_feature_name]
    batch = tf.io.decode_raw(batch, tf.dtypes.uint8)
    batch = tf.reshape(batch, batch_shape)

    # Apply augmentation with flips and rotations
    which_aug = tf.random.uniform(shape=(), dtype=tf.dtypes.int32, minval=0, maxval=8)
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


def load_dataset(input_context=None):
    """
    Load a dataset suitable for unsupervised training in a distributed environment:
        - If `input_context` is provided, it is used to build a single-replica dataset.
        - If `input_context` is not provided, builds a cross-replica dataset.

    In some cases (e.g. multi-GPU) a single dataset can be copied to multiple devices,
    so a cross-replica dataset should be built. This is also the default for
    single-device training. In other cases (e.g. TPUs or clusters) each device
    must build its own single-replica dataset by calling this function.

    For most purposes it suffices to call this function with `input_context=None`.
    Do not manually pass an input context; instead create per-replica datasets with
    `tf.distribute.experimental_distribute_datasets_from_function`.

    Parameters
    ----------
    input_context : None or tf.distribute.InputContext, optional
        If None, build a cross-replica, whole-dataset pipeline, as normal.
        If provided, shard the dataset to a single replica device.

    Returns
    -------
    tf.data.Dataset
        A dataset of batched examples, where each example is a set of coterminous
        input bands.
    """

    # If input_context is not provided, the dataset will be built once and copied to
    # each replica. Sharding and distribution will be handled automatically by the
    # distribution strategy, so we batch by global batch size.
    if input_context is not None and input_context.num_input_pipelines > 1:
        shard_data = True
        batch_size = input_context.get_per_replica_batch_size(FLAGS.batch_size)

    # Otherwise, _this function_ is copied to each replica and called once independently
    # per copy. The function should return an appropriate dataset for the replica --
    # manually sharded to be non-overlapping with other replicas and batched to
    # the replica batch size.
    else:
        shard_data = False
        batch_size = FLAGS.batch_size

    if FLAGS.data_file:
        dataset = _load_tfrecord(FLAGS.data_file)

        if shard_data:  # Shard granularity: lines of tfrecords
            dataset = dataset.shard(
                input_context.num_input_pipelines, input_context.input_pipeline_id
            )

        dataset = dataset.repeat()
    else:
        dataset = tf.data.experimental.CsvDataset(
            FLAGS.data_listing, [tf.dtypes.string]
        )

        if shard_data:  # Shard granularity: full tfrecord files
            dataset = dataset.shard(
                input_context.num_input_pipelines, input_context.input_pipeline_id
            )

        dataset = dataset.repeat().interleave(
            _load_tfrecord,
            cycle_length=FLAGS.tfrecord_parallel_reads,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

    if FLAGS.shuffle_buffer_size > 0:
        dataset = dataset.shuffle(
            FLAGS.shuffle_buffer_size,
            reshuffle_each_iteration=True,
            seed=FLAGS.random_seed,
        )

    dataset = (
        dataset.batch(batch_size, drop_remainder=True)
        .map(_preprocess_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    return dataset
