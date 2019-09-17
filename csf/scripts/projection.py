import sys

from absl import app, flags

from csf.experiments.projection import make_projection_figures


def main(_):
    make_projection_figures()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        flags.FLAGS(
            ["__main__", "--flagfile=csf/parameters/experiments_projection.cfg"]
        )
    app.run(main)
