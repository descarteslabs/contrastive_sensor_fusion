import sys

from absl import app, flags

from csf.experiments.classification import degrading_inputs_experiment


def main(_):
    degrading_inputs_experiment()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        flags.FLAGS(
            ["__main__", "--flagfile=csf/parameters/experiments_classification.cfg"]
        )
    app.run(main)
