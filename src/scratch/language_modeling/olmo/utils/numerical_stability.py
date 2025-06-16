"""Utility functions for numerical stability."""

import jax.numpy as jnp


def ensure_finite(
    x: jnp.ndarray, *, check_neg_inf: bool = True, check_pos_inf: bool = False
):
    """Ensure that x is finite.

    This is done by replacing float("-inf") with the minimum value of the dtype when
    check_neg_inf is True and by replacing float("inf") with the maximum value of the
    dtype when check_pos_inf is True.

    Args:
      x: Input tensor.
      check_neg_inf: Whether to check for negative infinity.
      check_pos_inf: Whether to check for positive infinity.

    Returns:
      The input tensor with the infinite values replaced.
    """

    def replace_inf(x, value):
        return jnp.where(jnp.isinf(x), value, x)

    if check_neg_inf:
        x = replace_inf(x, jnp.finfo(x.dtype).min)
    if check_pos_inf:
        x = replace_inf(x, jnp.finfo(x.dtype).max)
    return x
