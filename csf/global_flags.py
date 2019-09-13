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

flags.DEFINE_integer(
    "random_seed", 1337, "Random seed used for all training and experiments."
)


def n_bands():
    return len(FLAGS.bands)
