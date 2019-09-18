import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from absl import flags
from tensorflow.keras.layers import Concatenate, Input, Lambda

import csf.global_flags as gf
from csf.encoder import resnet_encoder

FLAGS = flags.FLAGS

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
    'PHR_nir',
)

def encoder_head(
    size,
    bands=None,
    batchsize=8,
    checkpoint_dir=None,
    checkpoint_file=None,
    trainable=False
):
    """Useful for building a model on top of the resnet encoder.
    Adds some convenience layers to reorder bands, provide zeros
    for missing bands, and to scale up bands based on missing rate.
    Returns 1. the overall model inputs, 2. the encoder inputs, which
    you may choose to ignore, and 3. the encoder outputs."""

    if bands is None:
        bands = FLAGS.bands

    n_bands = len(bands)

    if n_bands <= 0:
        raise ValueError("You must provide some bands")

    # Load upstream weights into encoder
    if checkpoint_dir == 'imagenet' or checkpoint_file == 'imagenet':
        n_input_bands = 3
        encoder = resnet_encoder(n_input_bands, weights='imagenet')
    else:
        n_input_bands = 12
        encoder = resnet_encoder(n_input_bands)
        if checkpoint_dir is not None:
            weights_path = tf.train.latest_checkpoint(checkpoint_dir)
            checkpoint = tf.train.Checkpoint(encoder=encoder)
            checkpoint.restore(weights_path).expect_partial()
        elif checkpoint_file is not None:
            checkpoint = tf.train.Checkpoint(encoder=encoder)
            checkpoint.restore(checkpoint_file).expect_partial()
    encoder.trainable = trainable

    model_inputs = Input(batch_shape=(batchsize, size, size, n_bands))

    # Reorganize the present inputs according to the order given
    to_concat = list()
    if n_input_bands == 12:
        for default_band in default_bands:
            try:
                band_i = bands.index(default_band)
            except ValueError:
                to_concat.append(K.zeros(shape=(batchsize, size, size, 1)))
            else:
                to_concat.append(K.expand_dims(model_inputs[..., band_i], axis=-1))
    else:
        for rgb_band in ('red', 'green', 'blue'):
            for band_i, band in enumerate(bands):
                if band.endswith(rgb_band):
                    to_concat.append(K.expand_dims(model_inputs[..., band_i], axis=-1))
                    break
            else:
                to_concat.append(K.zeros(shape=(batchsize, size, size, 1)))
    all_inputs = Concatenate(axis=-1)(to_concat)

    # Multiply inputs according to missing bands
    scaled_inputs = Lambda(lambda x: x * gf.n_bands() / n_bands)(all_inputs)
    encoded = encoder(scaled_inputs)

    return model_inputs, scaled_inputs, encoded


class LRMultiplierAdam(Adam):
    """Adam optimizer with varying per-layer learning rates.
    Parameters
    ----------
    multipliers : dict of layername (str) -> multipler (float)
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
