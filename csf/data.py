"""
Utilities to load data for the unsupervised learning task.
"""

import tensorflow as tf
from absl import flags

import csf.distribution
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
flags.DEFINE_bool(
    "enable_augmentation",
    False,
    "Whether to enable streaming augmentation (rotations and flips).",
)
flags.DEFINE_bool(
    "enable_experimental_optimization",
    True,
    "Whether to enable experimental optimizations.",
)
flags.DEFINE_integer(
    "shuffle_buffer_size",
    8000,
    "Number of examples to hold in the shuffle buffer. If <= 0, do not shuffle data.",
)
flags.DEFINE_integer(
    "tfrecord_parallel_reads", 16, "Number of tfrecord files to read in parallel."
)
flags.DEFINE_integer(
    "tfrecord_sequential_reads",
    16,
    "Number of exampls to read sequentially from each tfrecord.",
)
flags.DEFINE_integer("prefetch_batches", 32, "Number of batches to prefetch.")


def data_shape():
    """Get the shape of a single batch of input imagery."""
    return (
        csf.distribution.replica_batch_size(),
        FLAGS.data_tilesize,
        FLAGS.data_tilesize,
        gf.n_bands(),
    )


def load_dataset(input_context=None):
    """
    Load a dataset suitable for unsupervised training in a distributed environment:
        - If `input_context` is provided, it is used to build a single-replica dataset.
        - If `input_context` is not provided, builds a cross-replica dataset.

    In some cases (e.g. multi-GPU) a single dataset can be copied to multiple devices,
    so a cross-replica dataset should be built. This is also the default for
    single-device training. In other cases (e.g. TPU pods or clusters) each device
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
        input bands. Examples are flattened for efficient communication with hardware.
    """
    image_dims = (FLAGS.data_tilesize ** 2) * gf.n_bands()

    # NOTE: Make sure to test and fine-tune these optimizations for any new hardware.
    options = tf.data.Options()
    options.experimental_deterministic = False
    if FLAGS.enable_experimental_optimization:
        options.experimental_optimization.autotune_buffers = True
        options.experimental_optimization.autotune_cpu_budget = True
        options.experimental_optimization.parallel_batch = True
        options.experimental_optimization.map_fusion = True
        options.experimental_optimization.map_vectorization.enabled = False
        options.experimental_slack = True

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
        dataset = tf.data.TFRecordDataset(FLAGS.data_file).with_options(options)

        if shard_data:  # Shard granularity: lines of tfrecords
            dataset = dataset.shard(
                input_context.num_input_pipelines, input_context.input_pipeline_id
            )

        dataset = dataset.repeat()
    else:
        dataset = (
            tf.data.experimental.CsvDataset(FLAGS.data_listing, [tf.dtypes.string])
            .with_options(options)
            .map(
                lambda x: tf.reshape(x, []),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        )

        if shard_data:  # Shard granularity: full tfrecord files
            dataset = dataset.shard(
                input_context.num_input_pipelines, input_context.input_pipeline_id
            )

        dataset = dataset.repeat().interleave(
            tf.data.TFRecordDataset,
            cycle_length=FLAGS.tfrecord_parallel_reads,
            block_length=FLAGS.tfrecord_sequential_reads,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

    feature_spec = {
        FLAGS.data_feature_name: tf.io.FixedLenFeature([], tf.dtypes.string)
    }

    def preprocess(batch):
        batch = tf.io.parse_example(batch, feature_spec)
        batch = tf.io.decode_raw(
            batch[FLAGS.data_feature_name], tf.dtypes.uint8, fixed_length=image_dims
        )
        batch = tf.cast(batch, tf.dtypes.float32)
        batch = (batch - 128.0) / 128.0
        return batch

    def augment(batch):
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
        return batch

    if FLAGS.shuffle_buffer_size > 0:
        dataset = dataset.shuffle(
            FLAGS.shuffle_buffer_size,
            reshuffle_each_iteration=True,
            seed=FLAGS.random_seed,
        )

    # NOTE: examples are provided "flattened" and must be reshaped into images
    dataset = dataset.batch(batch_size, drop_remainder=True).map(
        preprocess, tf.data.experimental.AUTOTUNE
    )

    if FLAGS.enable_augmentation:
        dataset = dataset.map(augment, tf.data.experimental.AUTOTUNE)

    return dataset.prefetch(FLAGS.prefetch_batches)
