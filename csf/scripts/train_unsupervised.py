"""
Script which carries out unsupervised training. Configured by command-line arguments;
run `python3 csf/scripts/train_unsupervised --helpfull` for a complete list.
"""

from absl import app

from csf.train import run_unsupervised_training


def main(_):
    run_unsupervised_training()


if __name__ == "__main__":
    app.run(main)
