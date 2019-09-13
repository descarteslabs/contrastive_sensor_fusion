"""
Code to build ResNet encoders.
"""

import tensorflow as tf

# Layers of a ResNet50V2 model we tap into to get representations
# Reference input shape: (128, 128, 12)
RESNET_REPRESENTATION_LAYERS = [
    "conv2_block2_out",  # Reference output shape: (32, 32, 256)
    "conv3_block3_out",  # Reference output shape: (16, 16, 512)
    "conv4_block5_out",  # Reference output shape: (8,  8,  1024)
    "conv5_block3_out",  # Reference output shape: (4,  4,  2048)
]


def resnet_encoder(n_input_bands, weights=None):
    """
    Build a ResNet50V2 encoder. Takes input in the range [-1, 1].

    Parameters
    ----------
    input_shape : tuple
        Shape to use for the model's input.

    Returns
    -------
    tf.keras.Model
        A Model with a single tensor input and a dictionary of outputs,
        one for each activation of a residual stack.
    """
    model_base = tf.keras.applications.ResNet50V2(
        input_shape=input_shape, include_top=False, weights=weights, pooling=None
    )
    out_tensors = {
        layer: model_base.get_layer(layer).output
        for layer in RESNET_REPRESENTATION_LAYERS
    }
    model = tf.keras.Model(inputs=model_base.input, outputs=out_tensors)
    return model
