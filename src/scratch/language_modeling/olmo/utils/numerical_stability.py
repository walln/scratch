"""Utility functions for numerical stability."""

import torch


def ensure_finite(
    x: torch.Tensor, *, check_neg_inf: bool = True, check_pos_inf: bool = False
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
    if check_neg_inf:
        x.masked_fill_(x == float("-inf"), torch.finfo(x.dtype).min)
    if check_pos_inf:
        x.masked_fill_(x == float("inf"), torch.finfo(x.dtype).max)
