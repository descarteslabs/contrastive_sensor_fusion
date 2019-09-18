import numpy as np
import os.path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
assert tf.__version__.split('.', 1)[0] == '1', tf.__version__

import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Concatenate, Lambda
from lucid4keras import prepare_model, keras_render_vis, channel
import lucid.optvis.transform as transform
import lucid.optvis.param as param

param_f = lambda: param.image(128, fft=True, decorrelate=True)

transforms = [
    transform.pad(16),
    transform.jitter(8),
    transform.random_scale([n / 100.0 for n in range(80, 120)]),
    transform.random_rotate(list(range(-10, 11)) + list(range(-5, 6)) + list(range(-20, 21, 10))),
    transform.jitter(2)
]

batchsize = 1
size = 128

weightsfile = 'ckpt.h5'
if not os.path.isfile(weightsfile):
    raise RuntimeError("Run pre_visualize.py first, in a tensorflow 2.x "
                       "environment, to generate ckpt.h5")

# Load tf.keras weights into keras model for using lucid4keras.
model_base = keras.applications.ResNet50V2(
    input_shape=(size, size, 12),
    include_top=False,
    weights=None,
)
model_base.load_weights(weightsfile)

# We'll need to fill in extra bands we aren't using for RGB visualization.
BandFillLayer = Lambda(
    lambda x: K.stack([
        0 * x[...,0],
        0 * x[...,0],
        0 * x[...,0],
        0 * x[...,0],
        0 * x[...,0],
        0 * x[...,0],
        0 * x[...,0],
        0 * x[...,0],
        4.0 * x[...,0],
        4.0 * x[...,1],
        4.0 * x[...,2],
        0 * x[...,0]
    ], axis=-1)
)

# Made the model into a 3-band input for visualization
model_base_stack3 = prepare_model(model_base, layer_name="conv4_block5_out")
inputs_stack3 = Input(batch_shape=(batchsize, size, size, 3))
base_inputs_stack3 = BandFillLayer(inputs_stack3)
model_stack3 = Model(
    inputs=inputs_stack3,
    outputs=model_base_stack3(base_inputs_stack3)
)

try:
    print("Optimizing stack 3")
    for index in range(1024):
        objective = channel(index)
        im = keras_render_vis(model_stack3, objective, transforms=transforms, param_f=param_f, thresholds=[2560], lr=0.01)
        im = np.squeeze(im)
        im -= im.min(axis=(0, 1), keepdims=True)
        im /= im.max(axis=(0, 1), keepdims=True)
        plt.imshow(im)
        filename = 'vis_stack3_%04i.png' % (index,)
        print("Saving", filename)
        plt.savefig(filename)
        plt.clf()
except KeyboardInterrupt:
    pass

model_base_stack4 = prepare_model(model_base, layer_name="conv5_block3_out")
inputs_stack4 = Input(batch_shape=(batchsize, size, size, 3))
base_inputs_stack4 = BandFillLayer(inputs_stack4)
model_stack4 = Model(
    inputs=inputs_stack4,
    outputs=model_base_stack4(base_inputs_stack4)
)

print("Optimizing stack 4")
for index in range(2048):
    objective = channel(index)
    im = keras_render_vis(model_stack4, objective, transforms=transforms, param_f=param_f, thresholds=[2560], lr=0.02)
    im = np.squeeze(im)
    im -= im.min(axis=(0, 1), keepdims=True)
    im /= im.max(axis=(0, 1), keepdims=True)
    plt.imshow(im)
    filename = 'vis_stack4_%04i.png' % (index,)
    print("Saving", filename)
    plt.savefig(filename)
    plt.clf()
