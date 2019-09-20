"""
Code to build ResNet encoders.
"""

import tensorflow as tf
import tensorflow.keras.backend as K
from absl import flags
from tensorflow.keras.layers import Concatenate, Input, Lambda
from tensorflow.keras.optimizers import Adam

import csf.global_flags as gf

FLAGS = flags.FLAGS

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
    n_input_bands : int
        How many bands the model encodes.

    Returns
    -------
    tf.keras.Model
        A Model with a single tensor input and a dictionary of outputs,
        one for each activation of a residual stack.
    """
    model_base = tf.keras.applications.ResNet50V2(
        input_shape=(None, None, n_input_bands),
        include_top=False,
        weights=weights,
        pooling=None,
    )
    out_tensors = {
        layer: model_base.get_layer(layer).output
        for layer in RESNET_REPRESENTATION_LAYERS
    }
    return tf.keras.Model(inputs=model_base.input, outputs=out_tensors, name="encoder")


def encoder_head(
    size,
    bands=None,
    batch_size=8,
    checkpoint=None,
    trainable=True,
    assert_checkpoint=False,
):
    """
    Build a ResNet encoder from a subset of available bands, re-ordering the bands
    correctly and filling in any missing bands with zeroes.

    Parameters
    ----------
    size : int
        Tilesize this encoder accepts.
    bands : [string]
        List of bands this encoder uses. All other bands are filled in with zeroes.
        May appear in any order, but must be a subset of FLAGS.bands. If None,
        use all available bands.
    batch_size : int
        Batch size for the encoder.
    checkpoint : str
         - If "imagenet", the encoder is initialized with ImageNet weights and must
           have exactly 3 bands.
         - If a path to a checkpoint file (locally or in Google cloud), initialize
           from that checkpoint.
         - If a path to a directory containing checkpoint files (locally or in Google
           cloud), initialize from the latest checkpoint in that directory.
         - Otherwise, initialize randomly, or throw an error if `assert_checkpoint`
           is True.
    trainable : bool, optional
        If False, the encoder is frozen.
    assert_checkpoint : bool, optional
        If True, require that a checkpoint is loaded.

    Returns
    -------
    tf.Tensor
        The overall model inputs.
    tf.Tensor
        The encoder inputs.
    tf.Tensor
        The encoder outputs.
    """

    if bands is None:
        bands = FLAGS.bands
    n_bands = len(bands)

    if n_bands <= 0:
        raise ValueError("You must provide some bands.")

    # Load upstream weights into encoder
    if checkpoint == "imagenet":
        if n_bands != 3:
            raise ValueError(
                "If initializing an encoder with ImageNet weights, you"
                "must provide exactly three bands."
            )
        n_input_bands = 3
        encoder = resnet_encoder(3, weights="imagenet")
    else:
        n_input_bands = gf.n_bands()
        encoder = resnet_encoder(n_input_bands)

        ckpt = tf.train.Checkpoint(encoder=encoder)
        ckpt_path = tf.train.latest_checkpoint(checkpoint) or checkpoint
        ckpt_restore_status = ckpt.restore(ckpt_path)

        if assert_checkpoint:
            ckpt_restore_status.assert_nontrivial_match().expect_partial()
        else:
            ckpt_restore_status.expect_partial()

    encoder.trainable = trainable
    model_inputs = Input(batch_shape=(batch_size, size, size, n_bands))

    # Reorganize the present inputs according to the order given
    to_concat = list()
    if n_input_bands == 3:  # Order RGB bands correctly for ImageNet experiments
        for rgb_band in ("red", "green", "blue"):
            for band_i, band in enumerate(bands):
                if band.endswith(rgb_band):
                    to_concat.append(K.expand_dims(model_inputs[..., band_i], axis=-1))
                    break
            else:
                to_concat.append(K.zeros(shape=(batch_size, size, size, 1)))
    else:
        for default_band in FLAGS.bands:
            for band_i, band in enumerate(bands):
                if band == default_band:
                    to_concat.append(K.expand_dims(model_inputs[..., band_i], axis=-1))
                    break
            else:
                to_concat.append(K.zeros(shape=(batch_size, size, size, 1)))
    all_inputs = Concatenate(axis=-1)(to_concat)

    # Multiply inputs according to missing bands
    scaled_inputs = Lambda(lambda x: x * n_input_bands / n_bands)(all_inputs)
    encoded = encoder(scaled_inputs)

    return model_inputs, scaled_inputs, encoded


class LRMultiplierAdam(Adam):
    """
    Adam optimizer with varying per-layer learning rates.

    Parameters
    ----------
    multipliers : dict of layername (str) -> multipler (float)
        Per-layer learning rate multipliers.
    """

    def __init__(self, *args, multipliers={}, **kwargs):
        super(LRMultiplierAdam, self).__init__(*args, **kwargs)
        with K.name_scope(self.__class__.__name__):
            self.multipliers = {k: K.variable(v) for k, v in multipliers.items()}

    def get_updates(self, loss, params):
        # Mostly the same code as Adam class, with added multiplier variables.
        # Keras code from:
        # https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/keras/optimizers.py#L456
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (
                1.0 / (1.0 + self.decay * K.cast(self.iterations, K.dtype(self.decay)))
            )

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (
            K.sqrt(1.0 - K.pow(self.beta_2, t)) / (1.0 - K.pow(self.beta_1, t))
        )

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            layername = p.name.split("/", 1)[0]
            mult = self.multipliers.get(layername, 1.0)

            m_t = (self.beta_1 * m) + (1.0 - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1.0 - self.beta_2) * K.square(g)

            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - mult * lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - mult * lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, "constraint", None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates
