"""
Defines flags that are used project-wide.
"""

from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_list(
    "bands",
    None,
    "List of the names of the bands this model fuses, matching the unsupervised data.",
)
flags.mark_flag_as_required("bands")

flags.DEFINE_string(
    "tpu_address",
    None,
    "Address of the TPU to train with. Leave unspecified to train with other hardware.",
)


def n_bands():
    return len(FLAGS.bands)


def using_tpu():
    return FLAGS.tpu_address is not None
