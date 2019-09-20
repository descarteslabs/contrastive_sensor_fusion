import numpy as np
import os.path
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
import subprocess as sp
import time


def nn(representations, labels, ks):
    n_components = 2048
    pca = PCA(n_components)
    features = pca.fit_transform(representations)
    n_samples, n_channels = features.shape
    t0 = time.time()
    tree = cKDTree(features, leafsize=100)
    for k in ks:
        t0 = time.time()
        _, neighbor_indices = tree.query(features, k=(k + 1), n_jobs=-1, eps=0.0)
        neighbor_labels = labels[neighbor_indices]
        fraction_same = (np.sum(neighbor_labels == labels[...,np.newaxis]) - n_samples) / k
        print('  k=%03i: frac=%f' % (k, fraction_same / n_samples))


if __name__ == '__main__':
    paths = [
        '***REMOVED***',
        '***REMOVED***',
        '***REMOVED***',
        '***REMOVED***',
        '***REMOVED***',
        '***REMOVED***',
        '***REMOVED***',
        '***REMOVED***',
        '***REMOVED***',
        '***REMOVED***/'

        '***REMOVED***',
        '***REMOVED***',
        '***REMOVED***',

        '***REMOVED***',
        '***REMOVED***',
        '***REMOVED***',
        '***REMOVED***',
        '***REMOVED***',
        '***REMOVED***',
        '***REMOVED***',
        '***REMOVED***',
        '***REMOVED***',
        '***REMOVED***',
    ]
    for path in paths:
        dest_dir = path.replace('***REMOVED***', './')
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
        print(dest_dir)
        sp.check_call(('gsutil', '-mq', 'cp', '-n', os.path.join(path, '*'), dest_dir))
        features = np.load(os.path.join(dest_dir, 'representations.npy'))
        labels = np.load(os.path.join(dest_dir, 'labels.npy'))
        nn(features, labels, ks=[10, 25, 50, 100])
