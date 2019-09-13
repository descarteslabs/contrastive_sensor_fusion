import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Conv2D, Dense, Input, Lambda, GlobalMaxPooling2D
from tensorflow.keras import Model
import tensorflow.keras.backend as K

from csf.encoder import resnet_encoder

default_bands = (
    'SPOT_red',
    'SPOT_green',
    'SPOT_blue',
    'SPOT_nir',
    'NAIP_red',
    'NAIP_green',
    'NAIP_blue',
    'NAIP_nir',
    'PHR_red',
    'PHR_green',
    'PHR_blue',
    'PHR_nir'
)


def encoder_head(size, bands=None, batchsize=8, checkpoint_dir=None):
    """Useful for building a model on top of the resnet encoder.
    Adds some convenience layers to reorder bands, provide zeros
    for missing bands, and to scale up bands based on missing rate.
    Returns 1. the overall model inputs, 2. the encoder inputs, which
    you may choose to ignore, and 3. the encoder outputs."""

    if bands is None:
        bands = default_bands

    n_bands = len(bands)
    n_default_bands = len(default_bands)

    if n_bands <= 0:
        raise ValueError("You must provide some bands")

    encoder = resnet_encoder((size, size, 12))
    # Load upstream weights into encoder
    if checkpoint_dir is not None:
        weights_path = tf.train.latest_checkpoint(checkpoint_dir)
        checkpoint = tf.train.Checkpoint(encoder=encoder)
        checkpoint.restore(weights_path).expect_partial()
    encoder.trainable = False

    model_inputs = Input(batch_shape=(batchsize, size, size, n_bands))

    # Reorganize the present inputs according to the order given
    to_concat = list()
    bands_i = 0
    for default_band in default_bands:
        try:
            band_i = bands.index(default_band)
        except ValueError:
            to_concat.append(K.zeros(shape=(batchsize, size, size, 1)))
        else:
            to_concat.append(K.expand_dims(model_inputs[...,band_i], axis=-1))
    all_inputs = Concatenate(axis=-1)(to_concat)

    # Multiply inputs according to missing bands
    scaled_inputs = Lambda(lambda x: x * n_default_bands / n_bands)(all_inputs)
    encoded = encoder(scaled_inputs)

    return model_inputs, scaled_inputs, encoded
