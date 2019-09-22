"""
Code to find images that maximally activate certain output directions.
"""

import os
import pickle

import numpy as np
from absl import flags, logging
from PIL import Image
from sklearn.decomposition import PCA

from csf.experiments.projection import get_osm_representations
from csf.utils import maybe_make_path

flags.DEFINE_integer("num_units", 30, "Number of units to find salient images for.")
flags.DEFINE_integer("num_images", 10, "Number of images to find per unit.")
flags.DEFINE_float("image_size", 2.0, "Size of each image tile.")

FLAGS = flags.FLAGS


def save_salient_images():
    """
    Save the images that maximally activate certain output directions.
    """

    images_dir = os.path.join(FLAGS.out_dir, "salient_images")
    images_path = os.path.join(FLAGS.out_dir, "images.npy")
    labels_path = os.path.join(FLAGS.out_dir, "labels.npy")
    representations_path = os.path.join(FLAGS.out_dir, "representations.npy")
    pca_path = os.path.join(FLAGS.out_dir, "pca.pkl")

    maybe_make_path(FLAGS.out_dir)
    maybe_make_path(images_dir)

    if not all(map(os.path.exists, [images_path, labels_path, representations_path])):
        logging.info("Creating new representations.")
        images, labels, representations = get_osm_representations(FLAGS.checkpoint)
        np.save(os.path.join(FLAGS.out_dir, "images.npy"), images)
        np.save(os.path.join(FLAGS.out_dir, "labels.npy"), labels)
        np.save(os.path.join(FLAGS.out_dir, "representations.npy"), representations)
    else:
        logging.info("Loading existing representations.")

    images = np.load(images_path)
    representations = np.load(representations_path)

    if os.path.exists(pca_path):
        logging.info("Loading PCA from path: {}".format(pca_path))
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
    else:
        pca = PCA(n_components=FLAGS.num_units)
        pca.fit(representations)
        logging.info("Saving PCA to path: {}".format(pca_path))
        with open(pca_path, "wb") as f:
            pickle.dump(pca, f)

    logging.info("Running PCA with {} components".format(FLAGS.num_units))
    representations = pca.transform(representations)

    unit_indices = range(FLAGS.num_units)

    for i, unit in enumerate(unit_indices):
        indices = np.argsort(representations[:, unit])[-FLAGS.num_images :]
        imgs = images[indices]

        for image in range(FLAGS.num_images):
            image_path = os.path.join(images_dir, "{}_{}.png".format(i, image))
            Image.fromarray(imgs[image, :, :, :3]).save(image_path)
