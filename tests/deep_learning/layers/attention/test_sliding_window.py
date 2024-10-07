"""Test the sliding window mask for correctness."""

import jax.numpy as jnp

from scratch.deep_learning.layers.attention.multi_head_attention import (
    create_sliding_window_mask,
)


def test_sliding_window_mask():
    """Test the sliding window mask for correctness."""
    seq_len = 6
    window_size = 2
    mask = create_sliding_window_mask(seq_len, window_size, dtype=jnp.float32)

    expected_mask = jnp.array(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
        ],
        dtype=jnp.float32,
    )

    # Compare the generated mask with the expected mask
    assert jnp.allclose(mask[0, 0], expected_mask), "Sliding window mask is incorrect."
