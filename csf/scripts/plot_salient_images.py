"""
Plot a grid of the n OSM images which most strongly activate the first m units.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags, logging
from sklearn.decomposition import PCA

from csf.experiments.data import OSM_CLASSES
from csf.scripts import save_representations

flags.DEFINE_integer("num_units", 10, "Number of units to find salient images for.")
flags.DEFINE_integer("num_images", 10, "Number of images to find per unit.")
flags.DEFINE_integer(
    "pca_components", None, "If not None, instead plot principal components."
)
flags.DEFINE_float("image_size", 2.0, "Size of each image tile.")

FLAGS = flags.FLAGS


def main(args):
    if not os.path.exists(FLAGS.out_dir):
        logging.info("Creating path: {}".format(FLAGS.out_dir))
        os.makedirs(FLAGS.out_dir)

    images_path = os.path.join(FLAGS.out_dir, "images.npy")
    labels_path = os.path.join(FLAGS.out_dir, "labels.npy")
    representations_path = os.path.join(FLAGS.out_dir, "representations.npy")
    output_path = os.path.join(FLAGS.out_dir, "salient_images.png")

    if not all(map(os.path.exists, [images_path, labels_path, representations_path])):
        logging.info("Creating new representations.")
        save_representations.main(args)
    else:
        logging.info("Loading existing representations.")

    images = np.load(images_path)
    labels = np.load(labels_path)
    representations = np.load(representations_path)

    if FLAGS.pca_components:
        logging.info("Running PCA with {} components".format(FLAGS.pca_components))
        representations = PCA(n_components=FLAGS.pca_components).fit_transform(
            representations
        )

    plt.figure(
        figsize=(
            FLAGS.num_images * FLAGS.image_size,
            FLAGS.num_units * FLAGS.image_size,
        )
    )

    unit_indices = range(FLAGS.num_units)

    i = 1
    for unit in unit_indices:
        indices = np.argsort(representations[:, unit])[-FLAGS.num_images :]
        imgs = images[indices]
        lbls = labels[indices]

        for image in range(FLAGS.num_images):
            plt.subplot(FLAGS.num_units, FLAGS.num_images, i)
            plt.imshow(imgs[image, :, :, :3])
            plt.title("Label: {}".format(OSM_CLASSES[lbls[image]]))
            plt.axis("off")
            i += 1

    plt.savefig(output_path)


if __name__ == "__main__":
    app.run(main)
