"""
Utilities for distributed computation.
"""
from contextlib import contextmanager

import tensorflow as tf
from absl import flags, logging

import csf.data  # noqa

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "tpu",
    None,
    "Name or address of the TPU to train with. If unspecified, do not use TPU.",
)
flags.DEFINE_string(
    "tpu_zone",
    None,
    "Zone the TPU is located in. If unspecified, assume the same as the local VM.",
)
flags.DEFINE_bool(
    "run_distributed",
    False,
    "Enables multi-device training. On by default when a TPU is specified. "
    "Disables certain summaries.",
)
flags.DEFINE_bool(
    "multi_host",
    False,
    "Enables distributing ops across multiple hosts, as in a TPU pod.",
)

# Internal globals
_strategy = None
_initialized = False


@contextmanager
def _dummy_context():
    """A context manager that does nothing."""
    yield None


def _assert_initialized():
    if not _initialized:
        raise RuntimeError(
            "Distributed execution is not initialized. Call distribution.initialize()."
        )


def using_tpu():
    return FLAGS.tpu is not None


def num_replicas():
    """Get the number of synchronized replicas under the current strategy."""
    _assert_initialized()
    if FLAGS.run_distributed:
        return _strategy.num_replicas_in_sync
    return 1


def global_batch_size():
    """Get the batch size totaled across all replicas."""
    return FLAGS.batch_size


def replica_batch_size():
    """Get the batch size used in a single replica."""
    _assert_initialized()
    if not global_batch_size() % num_replicas() == 0:
        logging.warning(
            "Global batch size is not divisible by number of replicas. Rounding down."
        )
    return global_batch_size() // num_replicas()


def distribute_dataset_fn(dataset_fn):
    """
    Distribute a dataset under the current strategy.

    Parameters
    ----------
    dataset_fn : tf.distribute.InputContext -> tf.data.Dataset
        A function building a per-replica dataset.
        See csf.data.load_dataset for example.

    Returns
    -------
    tf.data.Dataset
        A distributed dataset producing per-replica batches.
    """
    _assert_initialized()

    if not FLAGS.run_distributed:  # Single-device training: use a single normal dataset
        logging.debug("Building single-device dataset.")
        return dataset_fn(None)

    if FLAGS.multi_host:  # Build one dataset per function
        logging.debug("Building per-replica datasets.")
        return _strategy.experimental_distribute_datasets_from_function(dataset_fn)

    # Multi-GPU or TPU training: build and distribute a single cross-replica dataset
    logging.debug("Building cross-replica dataset.")
    return _strategy.experimental_distribute_dataset(dataset_fn(None))


def distribute_computation(function):
    """
    Given a function to be replicated, return an operation which runs that function
    replicated across hardware in the distributed context.

    Parameters
    ----------
    function
        The function to be distributed. Must be autograph-compatible.

    Returns
    -------
    tf.Function
        The function, run through autograph and converted to run in the distributed
        context. The function takes a single argument, which is a list of args to pass
        to the wrapped function.
    """
    _assert_initialized()

    if FLAGS.run_distributed:

        def _wrapper(args):
            return _strategy.experimental_run_v2(function, args=args)

    else:

        def _wrapper(args):
            return function(*args)

    return _wrapper


def distributed_context():
    """
    A context manager that enters the scope of distributed computation, distributing
    variables and execution.
    """
    _assert_initialized()
    if FLAGS.run_distributed:
        return _strategy.scope()
    return _dummy_context()


def tpu_worker_context():
    """
    A context manager that places operations on the TPU worker thread if we're using
    a TPU, and otherwise does nothing.
    """
    _assert_initialized()
    if using_tpu():
        return tf.device("/job:worker")
    return _dummy_context()


def initialize():
    """
    Initialize hardware for distributed training. Must be called before any other
    distribution functions are.
    """
    global _strategy
    global _initialized

    if using_tpu():
        logging.info("Setting up TPU: {}.".format(FLAGS.tpu))
        FLAGS.run_distributed = True
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=FLAGS.tpu, zone=FLAGS.tpu_zone
        )
        tf.config.experimental_connect_to_host(cluster_resolver.master())
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        _strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
        logging.info("Done setting up TPU cluster.")
    elif FLAGS.run_distributed:
        logging.info("Setting up a mirrored distribution strategy.")
        _strategy = tf.distribute.MirroredStrategy()
    else:
        logging.info("Execution will not be distributed.")
        _strategy = None

    _initialized = True
    logging.info("Done initializing distributed execution.")
    logging.info("Replicas: {}.".format(num_replicas()))
    logging.info("Global batch size: {}.".format(global_batch_size()))
    logging.info("Replica batch size: {}.".format(replica_batch_size()))
