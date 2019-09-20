import sys

from absl import app, flags

from csf.experiments.projection import plot_osm_representations


def main(_):
    plot_osm_representations()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        flags.FLAGS(
            ["__main__", "--flagfile=csf/parameters/experiments_projection.cfg"]
        )
    app.run(main)
