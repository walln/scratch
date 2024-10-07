"""Tests for multi-head attention layers."""

import jax
import jax.numpy as jnp
from flax import nnx

from scratch.deep_learning.layers.attention import rope
from scratch.deep_learning.layers.attention.kv_cache import LayerKVCache
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
    assert attention_layer.head_dim == d_model // num_heads


def test_multi_head_attention_forward():
    """Test MultiHeadAttention forward pass."""
    d_model = 128
    num_heads = 8
    batch_size = 2
    seq_length = 10

    x = jnp.ones((batch_size, seq_length, d_model))

    attention_layer = MultiHeadAttention(d_model, num_heads, rngs=nnx.Rngs(0))
    output, _ = attention_layer(x)

    assert output.shape == (batch_size, seq_length, d_model)

    # Test with RoPE
    freqs_complex = rope.precompute_theta_pos_freqs(
        d_model // num_heads, seq_length * 2
    )[:seq_length]
    output_rope, _ = attention_layer(x, freqs_complex=freqs_complex)

    assert output_rope.shape == (batch_size, seq_length, d_model)


def test_multi_head_attention_with_mask():
    """Test MultiHeadAttention forward pass with mask."""
    d_model = 128
    num_heads = 8
    batch_size = 2
    seq_length = 10

    x = jnp.ones((batch_size, seq_length, d_model))
    mask = jnp.tril(jnp.ones((seq_length, seq_length)))

    attention_layer = MultiHeadAttention(d_model, num_heads, rngs=nnx.Rngs(0))
    output, _ = attention_layer(x, mask=mask)

    assert output.shape == (batch_size, seq_length, d_model)

    # Test with RoPE
    freqs_complex = rope.precompute_theta_pos_freqs(
        d_model // num_heads, seq_length * 2
    )[:seq_length]
    output_rope, _ = attention_layer(x, mask=mask, freqs_complex=freqs_complex)

    assert output_rope.shape == (batch_size, seq_length, d_model)


def test_multi_head_attention_with_kv_cache():
    """Test MultiHeadAttention forward pass with KV cache."""
    d_model = 128
    num_heads = 8
    batch_size = 2
    seq_length = 10

    x = jnp.ones((batch_size, seq_length, d_model))
    kv_cache = LayerKVCache.create(
        batch_size, seq_length, num_heads, d_model // num_heads
    )

    attention_layer = MultiHeadAttention(d_model, num_heads, rngs=nnx.Rngs(0))
    output, new_kv_cache = attention_layer(x, start_pos=0, kv_cache=kv_cache)

    assert output.shape == (batch_size, seq_length, d_model)
    assert new_kv_cache is not None

    # Test with RoPE
    freqs_complex = rope.precompute_theta_pos_freqs(
        d_model // num_heads, seq_length * 2
    )[:seq_length]
    output_rope, new_kv_cache_rope = attention_layer(
        x, start_pos=0, kv_cache=kv_cache, freqs_complex=freqs_complex
    )

    assert output_rope.shape == (batch_size, seq_length, d_model)
    assert new_kv_cache_rope is not None


def test_multi_head_attention_output_consistency():
    """Test if MultiHeadAttention output is consistent with and without KV cache."""
    d_model = 128
    num_heads = 8
    batch_size = 2
    seq_length = 10

    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_length, d_model))

    attention_layer = MultiHeadAttention(d_model, num_heads, rngs=nnx.Rngs(0))

    # Output without KV cache
    output_without_cache, _ = attention_layer(x)

    # Output with KV cache
    kv_cache = LayerKVCache.create(
        batch_size, seq_length, num_heads, d_model // num_heads
    )
    output_with_cache, _ = attention_layer(x, start_pos=0, kv_cache=kv_cache)

    # Check if outputs are close
    assert jnp.allclose(output_without_cache, output_with_cache, atol=1e-5)

    # Test with RoPE
    freqs_complex = rope.precompute_theta_pos_freqs(
        d_model // num_heads, seq_length * 2
    )[:seq_length]
    output_rope_without_cache, _ = attention_layer(x, freqs_complex=freqs_complex)
    output_rope_with_cache, _ = attention_layer(
        x, start_pos=0, kv_cache=kv_cache, freqs_complex=freqs_complex
    )

    assert jnp.allclose(output_rope_without_cache, output_rope_with_cache, atol=1e-5)


def test_sliding_window_attention():
    """Test MultiHeadAttention with sliding window."""
    d_model = 128
    num_heads = 8
    batch_size = 2
    seq_length = 10
    window_size = seq_length

    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_length, d_model))

    attention_layer = MultiHeadAttention(
        d_model, num_heads, sliding_window_size=window_size, rngs=nnx.Rngs(0)
    )
    output, _ = attention_layer(x)

    assert output.shape == (batch_size, seq_length, d_model)

    sliding_window_attention_layer = MultiHeadAttention(
        d_model, num_heads, sliding_window_size=window_size, rngs=nnx.Rngs(0)
    )
    sliding_output, _ = sliding_window_attention_layer(x)

    assert sliding_output.shape == (batch_size, seq_length, d_model)

    assert jnp.allclose(output, sliding_output, atol=1e-5)

    # Test with RoPE
    freqs_complex = rope.precompute_theta_pos_freqs(
        d_model // num_heads, seq_length * 2
    )[:seq_length]
    output_rope, _ = attention_layer(x, freqs_complex=freqs_complex)
    sliding_output_rope, _ = sliding_window_attention_layer(
        x, freqs_complex=freqs_complex
    )

    assert jnp.allclose(output_rope, sliding_output_rope, atol=1e-5)


def test_multi_head_attention_sliding_window_consistency():
    """Test if MultiHeadAttention output is consistent with/without sliding window."""
    d_model = 128
    num_heads = 8
    batch_size = 2
    seq_length = 10
    window_size = (
        seq_length  # Set window size to seq_length to match non-sliding behavior
    )

    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_length, d_model))
    freqs_complex = rope.precompute_theta_pos_freqs(
        d_model // num_heads, seq_length * 2
    )[:seq_length]

    # Attention layer without sliding window
    attention_layer_without_window = MultiHeadAttention(
        d_model, num_heads, rngs=nnx.Rngs(0)
    )

    # Test with RoPE
    output_without_window_rope, _ = attention_layer_without_window(
        x, freqs_complex=freqs_complex
    )

    # Test without RoPE
    output_without_window_no_rope, _ = attention_layer_without_window(x)

    # Attention layer with sliding window
    attention_layer_with_window = MultiHeadAttention(
        d_model, num_heads, sliding_window_size=window_size, rngs=nnx.Rngs(0)
    )

    # Test with RoPE
    output_with_window_rope, _ = attention_layer_with_window(
        x, freqs_complex=freqs_complex
    )

    # Test without RoPE
    output_with_window_no_rope, _ = attention_layer_with_window(x)

    # Check if outputs are close for both RoPE and non-RoPE cases
    assert jnp.allclose(output_without_window_rope, output_with_window_rope, atol=1e-5)
    assert jnp.allclose(
        output_without_window_no_rope, output_with_window_no_rope, atol=1e-5
    )
