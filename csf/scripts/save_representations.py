import os

import numpy as np
from absl import app, flags, logging

from csf.experiments.projection import get_osm_representations

flags.mark_flag_as_required("checkpoint")

FLAGS = flags.FLAGS


def main(_):
    if not os.path.exists(FLAGS.out_dir):
        logging.info("Creating path: {}".format(FLAGS.out_dir))
        os.makedirs(FLAGS.out_dir)

    images, labels, representations = get_osm_representations(FLAGS.checkpoint)
    np.save(os.path.join(FLAGS.out_dir, "images.npy"), images)
    np.save(os.path.join(FLAGS.out_dir, "labels.npy"), labels)
    np.save(os.path.join(FLAGS.out_dir, "representations.npy"), representations)


if __name__ == "__main__":
    app.run(main)
