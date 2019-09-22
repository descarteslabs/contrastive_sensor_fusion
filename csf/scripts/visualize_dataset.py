"""
Script which visualizes the dataset and view creation process in TensorBoard.

Configured by command-line arguments; run
`python3 csf/scripts/visualize_dataset --helpfull` for a complete list.
"""

import sys

from absl import app, flags

from csf.experiments.visualize_dataset import visualize_dataset


def main(_):
    visualize_dataset()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        flags.FLAGS(["__main__", "--flagfile=csf/parameters/training.cfg"])
    app.run(main)
