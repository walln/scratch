"""Masking utilities for local sliding window attention."""

import jax.numpy as jnp


def create_sliding_window_mask(
    seq_len: int, window_size: int, dtype=jnp.float32
) -> jnp.ndarray:
    """Creates a sliding window mask for attention.

    Args:
        seq_len: The length of the sequence.
        window_size: The size of the sliding window.
        dtype: The data type of the mask.

    Returns:
        A mask tensor of shape (1, 1, seq_len, seq_len) suitable for broadcasting.
    """
    # Create a matrix where each [i, j] is 1 if j is within the window of i
    mask = (
        jnp.abs(jnp.arange(seq_len).reshape(-1, 1) - jnp.arange(seq_len).reshape(1, -1))
        <= window_size
    )
    mask = mask.astype(dtype)
    # Expand dimensions to match attention score shape
    return mask[None, None, :, :]  # Shape: (1, 1, seq_len, seq_len)
