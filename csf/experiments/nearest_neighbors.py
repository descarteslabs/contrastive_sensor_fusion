import os.path
import subprocess as sp

import numpy as np
from absl import app, flags
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

from csf.experiments.projection import get_osm_representations

FLAGS = flags.FLAGS
K_VALUES = [10, 25, 50, 100]

flags.DEFINE_string(
    "save_path",
    None,
    "Path the representations and labels are saved at. "
    "If None (default), generate them from scratch.",
)
flags.DEFINE_integer(
    "pca_components",
    2048,
    "Number of components to use for PCA. before nearest-neighbor measurements.",
)


def nearest_neighbor_fraction_experiment(_):
    """
    For several values of k, print what fraction of the k nearest neighbors are the same
    class.
    """
    if FLAGS.save_path:
        features = np.load(os.path.join(FLAGS.save_path, "representations.npy"))
        labels = np.load(os.path.join(FLAGS.save_path, "labels.npy"))
    else:
        imgs, labels, features = get_osm_representations(FLAGS.checkpoint)
        del imgs

    pca = PCA(FLAGS.pca_components)
    features = pca.fit_transform(features)
    n_samples, n_channels = features.shape
    tree = cKDTree(features, leafsize=100)
    for k in K_VALUES:
        _, neighbor_indices = tree.query(features, k=(k + 1), n_jobs=-1, eps=0.0)
        neighbor_labels = labels[neighbor_indices]
        fraction_same = (
            np.sum(neighbor_labels == labels[..., np.newaxis]) - n_samples
        ) / k
        print("  k=%03i: frac=%f" % (k, fraction_same / n_samples))


if __name__ == "__main__":
    app.run(nearest_neighbor_fraction_experiment)
