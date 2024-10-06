"""Tests for dot-product attention layer."""

import jax
import jax.numpy as jnp

from scratch.deep_learning.layers.attention.dot_product_attention import (
    DotProductAttention,
)


def test_dot_product_attention_initialization():
    """Test initialization of DotProductAttention layer."""
    attention_layer = DotProductAttention()
    assert isinstance(attention_layer, DotProductAttention)


def test_dot_product_attention_forward():
    """Test forward pass of DotProductAttention layer."""
    input_dim = 64
    batch_size = 2
    seq_length = 10

    query = jnp.ones((batch_size, seq_length, input_dim))
    value = jnp.ones((batch_size, seq_length, input_dim))

    attention_layer = DotProductAttention()
    context, attention = attention_layer(query, value)

    assert context.shape == query.shape
    assert attention.shape == (batch_size, seq_length, seq_length)
    assert jnp.allclose(jnp.sum(attention, axis=-1), 1.0)


def test_dot_product_attention_zero_input():
    """Test DotProductAttention with zero input."""
    input_dim = 64
    batch_size = 2
    seq_length = 10

    query = jnp.zeros((batch_size, seq_length, input_dim))
    value = jnp.zeros((batch_size, seq_length, input_dim))

    attention_layer = DotProductAttention()
    context, attention = attention_layer(query, value)

    assert context.shape == query.shape
    assert attention.shape == (batch_size, seq_length, seq_length)
    assert jnp.allclose(context, 0.0)


def test_dot_product_attention_random_input():
    """Test DotProductAttention with random input."""
    input_dim = 64
    batch_size = 2
    seq_length = 10

    query = jnp.array(
        jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_length, input_dim))
    )
    value = jnp.array(
        jax.random.normal(jax.random.PRNGKey(1), (batch_size, seq_length, input_dim))
    )

    attention_layer = DotProductAttention()
    context, attention = attention_layer(query, value)

    assert context.shape == query.shape
    assert attention.shape == (batch_size, seq_length, seq_length)
    assert jnp.allclose(jnp.sum(attention, axis=-1), 1.0)
