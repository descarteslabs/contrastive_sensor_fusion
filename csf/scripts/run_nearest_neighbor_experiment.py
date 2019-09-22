"""
Script which runs a nearest-neighbor experiment.

Configured by command-line arguments; run
`python3 csf/scripts/run_nearest_neighbor_experiment --helpfull` for a complete list.
"""

import sys

from absl import app, flags

from csf.experiments.nearest_neighbors import nearest_neighbor_fraction_experiment


def main(_):
    if len(sys.argv) == 1:
        flags.FLAGS(["__main__", "--flagfile=csf/parameters/experiments.cfg"])
    nearest_neighbor_fraction_experiment()


if __name__ == "__main__":
    app.run(main)
