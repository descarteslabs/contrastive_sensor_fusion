"""Prepare TensorFlow 2.0 Keras encoder weights for TensorFlow 1.0 Keras."""

import matplotlib as mpl
import tensorflow as tf
from absl import app, flags

flags.DEFINE_string("checkpoint", None, "Checkpoint file or directory to port.")
flags.mark_flag_as_required("checkpoint")

FLAGS = flags.FLAGS


def main(_):
    mpl.use("Agg")

    assert tf.__version__.split(".", 1)[0] == "2", (tf.__version__, tf.__file__)

    weights_dest = "ckpt.h5"
    model_base = tf.keras.applications.ResNet50V2(
        input_shape=(128, 128, 12), include_top=False, weights=None
    )

    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint) or FLAGS.checkpoint
    checkpoint = tf.train.Checkpoint(encoder=model_base)
    checkpoint.restore(checkpoint_file).expect_partial()

    model_base.save_weights(weights_dest)


if __name__ == "__main__":
    app.run(main)
