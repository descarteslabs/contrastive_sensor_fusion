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
    int -> float tf.Tensor
        Function from step to value, interpolating in the defined range.
    """
    start_step = tf.cast(start_step, tf.dtypes.float32)
    end_step = tf.cast(end_step, tf.dtypes.float32)
    step = tf.cast(step, tf.dtypes.float32)
    step = tf.minimum(tf.maximum(step, start_step), end_step)
    return initial_value + (step - start_step) * (initial_value - final_value) / (
        start_step - end_step
    )
