"""
Script which plots a grid of the n OSM images which most strongly activate certain
directions in output space.

Configured by command-line arguments; run
`python3 csf/scripts/plot_salient_images --helpfull` for a complete list.
"""

import sys

from absl import app, flags

from csf.experiments.salient_images import save_salient_images


def main(_):
    save_salient_images()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        flags.FLAGS(["__main__", "--flagfile=csf/parameters/experiments.cfg"])
    app.run(main)
