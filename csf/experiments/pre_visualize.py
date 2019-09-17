import numpy as np
import os.path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
assert tf.__version__.split('.', 1)[0] == '2', (tf.__version__, tf.__file__)

batchsize = 1
size = 128 # 128 148 134 137 140

weightsfile = 'ckpt-80.h5'
model_base = tf.keras.applications.ResNet50V2(
    input_shape=(size, size, 12),
    include_top=False,
    weights=None,
)

checkpoint_file = '***REMOVED***outputs/basenets_fusion_tpu_deploy_1/ckpt-80'
checkpoint = tf.train.Checkpoint(encoder=model_base)
checkpoint.restore(checkpoint_file).expect_partial()

model_base.save_weights(weightsfile)
