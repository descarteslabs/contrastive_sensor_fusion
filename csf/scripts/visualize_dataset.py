"""
Code to visualize the dataset and view creation process in TensorBoard.
"""

from functools import partial

import tensorflow as tf
from absl import app, flags

import csf.data
import csf.train
import csf.utils  # noqa

flags.DEFINE_list(
    "visualize_bands", None, "Bands to visualize. Should be grouped into blocks of 3."
)
flags.mark_flag_as_required("visualize_bands")

flags.DEFINE_integer("max_pages", 100, "Maximum number of summary pages to create.")
flags.DEFINE_integer("images_per_page", 1, "Number of summaries on a single page.")
flags.DEFINE_integer("views", 2, "Number of views to show summaries for.")

FLAGS = flags.FLAGS


def main(_):
    FLAGS.batch_size = FLAGS.images_per_page  # Prevent loading more bands than we plot
    csf.distribution.initialize()
    dataset = csf.data.load_dataset()
    summary_writer = tf.summary.create_file_writer(FLAGS.out_dir)
    page = tf.Variable(0, trainable=False, name="step", dtype=tf.dtypes.int64)
    tf.summary.experimental.set_step(page)

    _visualize_batch = partial(
        csf.utils.visualize_batch,
        visualize_bands=FLAGS.visualize_bands,
        max_outputs=FLAGS.images_per_page,
    )

    for batch in dataset:
        batch = tf.reshape(batch, csf.data.data_shape())
        current_page = int(page.assign_add(1))
        if current_page > FLAGS.max_pages:
            break

        with tf.name_scope("input_imagery"), summary_writer.as_default():
            _visualize_batch(batch)
        for i in range(FLAGS.views):
            with tf.name_scope("view_{}".format(i)), summary_writer.as_default():
                view = csf.train._create_view(batch, FLAGS.band_dropout_rate, i)
                _visualize_batch(view)


if __name__ == "__main__":
    app.run(main)
