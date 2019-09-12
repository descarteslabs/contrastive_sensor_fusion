"""
Various utilities used project-wide.
"""

from contextlib import contextmanager

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


def partition_imagery(concatenated_imagery, image_bands, partition_bands):
    """
    Extract, by name, a list of bands from a tensor of concatenated imagery.
    Results are returned as a list of named 3-band images for visualization.

    Parameters
    ----------
    concatenated_imagery : tf.Tensor
        A rank-4 tensor. The first dimension is treated as a batch dimension, and the
        last dimension is treated as channels.
    image_bands : [string]
        Names of the bands in the input tensor. Must have length equal to the size of
        the last axis of `concatenated_imagery`.
    partition_bands : [string]
        Names of bands to extract. Length must be divisible by 3.

    Returns
    -------
    [string], [tf.Tensor]
        Strings are names for the resuting tensors, and tensors are 3-channel images,
        one for every 3 bands in `partition_bands`.
    """

    def extract_group(bands):
        indices = [image_bands.index(band) for band in bands]
        return tf.gather(concatenated_imagery, indices, axis=-1)

    n_groups = len(partition_bands) // 3
    groups = []
    for i in range(n_groups):
        bands = partition_bands[3 * i : 3 * (i + 1)]
        imagery = extract_group(bands)
        groups.append((",".join(bands), imagery))

    return list(zip(*groups))


@contextmanager
def dummy_context():
    """A context manager that does nothing."""
    yield None
