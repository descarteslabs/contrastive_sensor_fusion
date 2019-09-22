"""
Script which carries out unsupervised training.

Configured by command-line arguments; run
`python3 csf/scripts/train_unsupervised --helpfull` for a complete list.
"""

import sys

from absl import app, flags

from csf.train import run_unsupervised_training


def main(_):
    run_unsupervised_training()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        flags.FLAGS(["__main__", "--flagfile=csf/parameters/training.cfg"])
    app.run(main)
