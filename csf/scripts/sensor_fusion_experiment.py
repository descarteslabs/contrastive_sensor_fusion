import sys

from absl import app, flags

from csf.experiments.hypercolumns import sensior_fusion_experiment


def main(_):
    sensior_fusion_experiment()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        flags.FLAGS(
            ["__main__", "--flagfile=csf/parameters/experiments_hypercolumns.cfg"]
        )
    app.run(main)
