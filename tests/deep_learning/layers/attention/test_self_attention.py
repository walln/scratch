"""Tests for self-attention layer."""

import jax
import jax.numpy as jnp
from flax import nnx

from scratch.deep_learning.layers.attention.self_attention import SelfAttention


def test_self_attention_initialization():
    """Test initialization of SelfAttention layer."""
    input_dim = 64
    attention_layer = SelfAttention(input_dim, rngs=nnx.Rngs(0))
    assert attention_layer.input_dim == input_dim
    assert isinstance(attention_layer.key, nnx.Linear)
    assert isinstance(attention_layer.query, nnx.Linear)
    assert isinstance(attention_layer.value, nnx.Linear)


def test_self_attention_forward():
    """Test forward pass of SelfAttention layer."""
    input_dim = 64
    batch_size = 2
    seq_length = 10

    x = jnp.ones((batch_size, seq_length, input_dim))

    attention_layer = SelfAttention(input_dim, rngs=nnx.Rngs(0))
    output, attention = attention_layer(x)

    assert output.shape == x.shape
    assert attention.shape == (batch_size, seq_length, seq_length)
    assert jnp.allclose(jnp.sum(attention, axis=-1), 1.0)


def test_self_attention_zero_input():
    """Test SelfAttention with zero input."""
    input_dim = 64
    batch_size = 2
    seq_length = 10

    x = jnp.zeros((batch_size, seq_length, input_dim))

    attention_layer = SelfAttention(input_dim, rngs=nnx.Rngs(0))
    output, attention = attention_layer(x)

    assert output.shape == x.shape
    assert attention.shape == (batch_size, seq_length, seq_length)
    assert jnp.allclose(output, 0.0)


def test_self_attention_random_input():
    """Test SelfAttention with random input."""
    input_dim = 64
    batch_size = 2
    seq_length = 10

    x = jnp.array(
        jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_length, input_dim))
    )

    attention_layer = SelfAttention(input_dim, rngs=nnx.Rngs(0))
    output, attention = attention_layer(x)

    assert output.shape == x.shape
    assert attention.shape == (batch_size, seq_length, seq_length)
    assert jnp.allclose(jnp.sum(attention, axis=-1), 1.0)
