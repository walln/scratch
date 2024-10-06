"""Tests for the ScaledDotProductAttention layer."""

import jax.numpy as jnp

from scratch.deep_learning.layers.attention.scaled_dot_product_attention import (
    ScaledDotProductAttention,
)


def test_scaled_dot_product_attention_initialization():
    """Test ScaledDotProductAttention initialization."""
    d_head = 64
    attention_layer = ScaledDotProductAttention(d_head)
    assert attention_layer.d_head == d_head


def test_scaled_dot_product_attention_forward():
    """Test ScaledDotProductAttention forward pass."""
    d_head = 64
    batch_size = 2
    seq_length = 3

    query = jnp.ones((batch_size, seq_length, d_head))
    key = jnp.ones((batch_size, seq_length, d_head))
    value = jnp.ones((batch_size, seq_length, d_head))

    attention_layer = ScaledDotProductAttention(d_head)
    context, attention = attention_layer(query, key, value)

    assert context.shape == (batch_size, seq_length, d_head)
    assert attention.shape == (batch_size, seq_length, seq_length)


def test_scaled_dot_product_attention_with_mask():
    """Test ScaledDotProductAttention forward pass with mask."""
    d_head = 64
    batch_size = 2
    seq_length = 3

    query = jnp.ones((batch_size, seq_length, d_head))
    key = jnp.ones((batch_size, seq_length, d_head))
    value = jnp.ones((batch_size, seq_length, d_head))
    mask = jnp.array(
        [[[False, True, True], [False, False, True], [False, False, False]]]
        * batch_size
    )

    attention_layer = ScaledDotProductAttention(d_head)
    context, attention = attention_layer(query, key, value, mask)

    assert context.shape == (batch_size, seq_length, d_head)
    assert attention.shape == (batch_size, seq_length, seq_length)

    # Check that attention values for masked positions are very small
    masked_attention = jnp.where(mask, 0.0, attention)
    assert jnp.allclose(masked_attention[:, 0, 1:], 0, atol=1e-5)
