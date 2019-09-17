"""Prepare tensorflow 2.0 keras encoder weights for tensorflow 1.0 keras"""

import numpy as np
import os.path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
assert tf.__version__.split('.', 1)[0] == '2', (tf.__version__, tf.__file__)

batchsize = 1
size = 128

weights_dest = 'ckpt.h5'
model_base = tf.keras.applications.ResNet50V2(
    input_shape=(size, size, 12),
    include_top=False,
    weights=None,
)

checkpoint_file = tf.train.latest_checkpoint('gs://dl-appsci/basenets/outputs/basenets_fusion_tpu_deploy_2/')
checkpoint = tf.train.Checkpoint(encoder=model_base)
checkpoint.restore(checkpoint_file).expect_partial()

model_base.save_weights(weights_dest)
