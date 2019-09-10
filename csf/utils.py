"""
Various utilities used project-wide.
"""

import tensorflow as tf


def linear_interpolate(step, initial_value, final_value, start_step, end_step):
    """
    Linearly interpolate between two values in a range of time steps.
    Values before the range of steps are fixed at `initial_value`, and values after
    the range are fixed at `final_value.`

    Parameters
    ----------
    step : number or tf.Tensor
        The step to use as input to interpolation.
    initial_value : number or tf.Tensor
        The first value used for interpolation. Must be compatible with float32.
    final_value : number or tf.Tensor
        The second value used for interpolation. Must be compatible with float32.
    start_step : number or tf.Tensor
        First step in the range of interpolated values. Must be compatible with float32.
    end_step : number or tf.Tensor
        Last step in the range of interpolated values. Must be compatible with float32.

    Returns
    -------
    tf.Tensor
        Float32 tensor containing the value interpolated in the range.
    """
    start_step = tf.cast(start_step, tf.dtypes.float32)
    end_step = tf.cast(end_step, tf.dtypes.float32)
    step = tf.cast(step, tf.dtypes.float32)
    step = tf.minimum(tf.maximum(step, start_step), end_step)
    return initial_value + (step - start_step) * (initial_value - final_value) / (
        start_step - end_step
    )


def optional_warmup(step, final_value, warmup_steps):
    """
    Warm a value up to a final value over a number of steps, or keep the value constant
    if no warmup range is defined. Used to schedule various hyperparameters.

    Parameters
    ----------
    step : number or tf.Tensor
        The step to use as input to interpolation.
    final_value : number or tf.Tensor
        The value to warm up to. Must be compatible with float32.
    warmup_steps : None or number
        Number of steps to scale value over. If None, value is not scaled.

    Returns
    -------
    tf.Tensor
        Float32 tensor containing the value interpolated in the range.
    """
    if warmup_steps:
        return linear_interpolate(step, 0.0, final_value, 0, warmup_steps)

    return tf.cast(final_value, dtype=tf.dtypes.float32)


def make_legal_image_summary(tensor):
    """
    Make a rank-4 tensor into a legal tensor for image summary by truncating or padding
    channels.

    Parameters
    ----------
    tensor : tf.Tensor
        A rank-4 tensor. The first dimension is treated as a batch dimension, and the
        last dimension is treated as channels.

    Returns
    -------
    tf.Tensor
        A rank-4 tensor with 1 or 3 channels, for visualization with tf.summary.image.
    """
    n_channels = tensor.shape.as_list()[-1]
    if n_channels == 1 or n_channels == 3:
        return tensor
    if n_channels > 3:
        return tensor[:, :, :, :3]
    return tf.pad(tensor, ((0, 0), (0, 0), (0, 0), (0, 3 - n_channels)))
