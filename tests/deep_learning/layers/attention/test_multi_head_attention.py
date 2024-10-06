"""Tests for multi-head attention layers."""

import jax
import jax.numpy as jnp
from flax import nnx

from scratch.deep_learning.layers.attention.multi_head_attention import (
    MultiHeadAttention,
)


def test_multi_head_attention_initialization():
    """Test MultiHeadAttention initialization."""
    d_model = 128
    num_heads = 8
    attention_layer = MultiHeadAttention(d_model, num_heads, rngs=nnx.Rngs(0))
    assert attention_layer.d_model == d_model
    assert attention_layer.num_heads == num_heads
    assert attention_layer.d_head == d_model // num_heads


def test_multi_head_attention_forward():
    """Test MultiHeadAttention forward pass."""
    d_model = 128
    num_heads = 8
    batch_size = 2
    seq_length = 10

    query = jnp.ones((batch_size, seq_length, d_model))
    key = jnp.ones((batch_size, seq_length, d_model))
    value = jnp.ones((batch_size, seq_length, d_model))

    attention_layer = MultiHeadAttention(d_model, num_heads, rngs=nnx.Rngs(0))
    output, _ = attention_layer(query, key, value)

    assert output.shape == (batch_size, seq_length, d_model)


def test_multi_head_attention_with_mask():
    """Test MultiHeadAttention forward pass with mask."""
    d_model = 128
    num_heads = 8
    batch_size = 2
    seq_length = 10

    query = jnp.ones((batch_size, seq_length, d_model))
    key = jnp.ones((batch_size, seq_length, d_model))
    value = jnp.ones((batch_size, seq_length, d_model))
    mask = jnp.array(
        [[[False, True, True, True, True, True, True, True, True, True]] * seq_length]
        * batch_size
    )

    attention_layer = MultiHeadAttention(d_model, num_heads, rngs=nnx.Rngs(0))
    output, _ = attention_layer(query, key, value, mask)

    assert output.shape == (batch_size, seq_length, d_model)


def test_attention_values_validity():
    """Test if attention values are within the valid range [0, 1]."""
    batch_size, seq_length, d_model, n_heads = 2, 16, 512, 8

    # Random input tensor
    x = jax.random.uniform(jax.random.PRNGKey(0), (batch_size, seq_length, d_model))
    mask = jnp.zeros((batch_size, seq_length, seq_length))
    mask = jnp.tril(mask)

    # Initialize MultiHeadAttention layer
    mha = MultiHeadAttention(d_model, n_heads, rngs=nnx.Rngs(0))

    # Forward pass
    _, attention = mha(x, x, x, mask)

    # Check attention values are within the valid range [0, 1]
    assert jnp.all(attention >= 0), "Attention values must be non-negative."
    assert jnp.all(attention <= 1), "Attention values must be less than or equal to 1."

    # Check if the sum of the attention weights for each query across all keys is 1
    attention_sum = attention.sum(axis=-1)
    assert jnp.allclose(
        attention_sum, jnp.ones_like(attention_sum)
    ), "Sum of attention weights must be 1."
