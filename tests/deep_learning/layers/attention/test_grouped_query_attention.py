"""Tests for the grouped query attention module."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from scratch.deep_learning.layers.attention import rope
from scratch.deep_learning.layers.attention.grouped_query_attention import (
    GroupedQueryAttention,
)
from scratch.deep_learning.layers.attention.kv_cache import LayerKVCache


def create_gqa_module(d_model, n_heads, n_kv_heads, rngs=None):
    """Create a grouped query attention module."""
    return GroupedQueryAttention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        rngs=rngs or nnx.Rngs(0),
    )


def test_output_shape():
    """Test that the output shape of the grouped query attention module is correct."""
    d_model, n_heads, n_kv_heads = 512, 8, 2
    batch_size, seq_len = 2, 16

    gqa_module = create_gqa_module(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))
    freqs_complex = rope.precompute_theta_pos_freqs(d_model // n_heads, seq_len * 2)[
        :seq_len
    ]

    # Test with RoPE
    output_rope, _ = gqa_module(x, freqs_complex=freqs_complex)
    assert output_rope.shape == (batch_size, seq_len, d_model)

    # Test without RoPE
    output_no_rope, _ = gqa_module(x)
    assert output_no_rope.shape == (batch_size, seq_len, d_model)


def test_kv_cache_update():
    """Test that the KV cache is updated correctly."""
    d_model, n_heads, n_kv_heads = 512, 8, 2
    batch_size, seq_len = 2, 16

    gqa_module = create_gqa_module(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))
    freqs_complex = rope.precompute_theta_pos_freqs(d_model // n_heads, seq_len * 2)[
        :seq_len
    ]

    kv_cache = LayerKVCache.create(batch_size, seq_len, n_kv_heads, d_model // n_heads)

    # Test with RoPE
    _, new_kv_cache_rope = gqa_module(x, freqs_complex=freqs_complex, kv_cache=kv_cache)
    assert new_kv_cache_rope is not None
    assert jnp.any(new_kv_cache_rope.k != 0)
    assert jnp.any(new_kv_cache_rope.v != 0)

    # Test without RoPE
    _, new_kv_cache_no_rope = gqa_module(x, kv_cache=kv_cache)
    assert new_kv_cache_no_rope is not None
    assert jnp.any(new_kv_cache_no_rope.k != 0)
    assert jnp.any(new_kv_cache_no_rope.v != 0)


def test_no_kv_cache():
    """Test that the KV cache is not used when not provided."""
    d_model, n_heads, n_kv_heads = 512, 8, 2
    batch_size, seq_len = 2, 16

    gqa_module = create_gqa_module(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))
    freqs_complex = rope.precompute_theta_pos_freqs(d_model // n_heads, seq_len * 2)[
        :seq_len
    ]

    # Test with RoPE
    output_rope, kv_cache_rope = gqa_module(x, freqs_complex=freqs_complex)
    assert output_rope.shape == (batch_size, seq_len, d_model)
    assert kv_cache_rope is None

    # Test without RoPE
    output_no_rope, kv_cache_no_rope = gqa_module(x)
    assert output_no_rope.shape == (batch_size, seq_len, d_model)
    assert kv_cache_no_rope is None


@pytest.mark.parametrize(
    ("d_model", "n_heads", "n_kv_heads"),
    [(512, 8, 2), (512, 8, 4), (768, 12, 3), (1024, 16, 4)],
)
def test_different_head_configurations(d_model: int, n_heads: int, n_kv_heads: int):
    """Test that the module works with different head configurations."""
    batch_size, seq_len = 2, 16

    gqa_module = create_gqa_module(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))
    freqs_complex = rope.precompute_theta_pos_freqs(d_model // n_heads, seq_len * 2)[
        :seq_len
    ]

    # Test with RoPE
    output_rope, _ = gqa_module(x, freqs_complex=freqs_complex)
    assert output_rope.shape == (batch_size, seq_len, d_model)

    # Test without RoPE
    output_no_rope, _ = gqa_module(x)
    assert output_no_rope.shape == (batch_size, seq_len, d_model)


@pytest.mark.parametrize(
    ("d_model", "n_heads", "n_kv_heads"),
    [
        (512, 8, 3),  # n_heads not divisible by n_kv_heads
        (512, 7, 1),  # d_model not divisible by n_heads
        (768, 12, 5),  # n_heads not divisible by n_kv_heads
    ],
)
def test_invalid_head_configuration(d_model: int, n_heads: int, n_kv_heads: int):
    """Test that an assertion error is raised when the head configuration is invalid."""
    with pytest.raises(AssertionError):
        create_gqa_module(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
        )


def test_forward_backward():
    """Test the forward and backward pass of the grouped query attention module."""
    d_model, n_heads, n_kv_heads = 512, 8, 2
    batch_size, seq_len = 2, 16

    gqa_module = create_gqa_module(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))
    freqs_complex = rope.precompute_theta_pos_freqs(d_model // n_heads, seq_len * 2)[
        :seq_len
    ]

    # Test with RoPE
    def loss_fn_rope(model):
        output, _ = model(x, freqs_complex=freqs_complex)
        return jnp.mean(output**2)

    loss_rope, grads_rope = nnx.value_and_grad(loss_fn_rope)(gqa_module)

    graphdef, params = nnx.split(gqa_module, nnx.Param)
    shape_check_rope = jax.tree_util.tree_map(
        lambda p, g: p.shape == g.shape, params, grads_rope
    )
    assert jax.tree_util.tree_all(shape_check_rope), (
        f"Shapes don't match with RoPE: {shape_check_rope}"
    )

    # Test without RoPE
    def loss_fn_no_rope(model):
        output, _ = model(x)
        return jnp.mean(output**2)

    loss_no_rope, grads_no_rope = nnx.value_and_grad(loss_fn_no_rope)(gqa_module)

    shape_check_no_rope = jax.tree_util.tree_map(
        lambda p, g: p.shape == g.shape, params, grads_no_rope
    )
    assert jax.tree_util.tree_all(shape_check_no_rope), (
        f"Shapes don't match without RoPE: {shape_check_no_rope}"
    )


@pytest.mark.parametrize("start_pos", [0, 8, 16])
def test_incremental_forward(start_pos):
    """Test that the grouped query attention module can be used incrementally."""
    d_model, n_heads, n_kv_heads = 512, 8, 2
    batch_size, seq_len = 2, 16

    gqa_module = create_gqa_module(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))
    freqs_complex = rope.precompute_theta_pos_freqs(
        d_model // n_heads, (start_pos + seq_len) * 2
    )[start_pos : start_pos + seq_len]

    kv_cache = LayerKVCache.create(
        batch_size, start_pos + seq_len, n_kv_heads, d_model // n_heads
    )

    # Test with RoPE
    output_rope, new_kv_cache_rope = gqa_module(
        x, freqs_complex=freqs_complex, start_pos=start_pos, kv_cache=kv_cache
    )
    assert output_rope.shape == (batch_size, seq_len, d_model)
    assert new_kv_cache_rope is not None
    assert jnp.any(new_kv_cache_rope.k[:, start_pos : start_pos + seq_len] != 0)

    # Test without RoPE
    output_no_rope, new_kv_cache_no_rope = gqa_module(
        x, start_pos=start_pos, kv_cache=kv_cache
    )
    assert output_no_rope.shape == (batch_size, seq_len, d_model)
    assert new_kv_cache_no_rope is not None
    assert jnp.any(new_kv_cache_no_rope.k[:, start_pos : start_pos + seq_len] != 0)


def test_numerical_stability():
    """Test that the grouped query attention module is numerically stable."""
    d_model, n_heads, n_kv_heads = 512, 8, 2
    batch_size, seq_len = 2, 16

    gqa_module = create_gqa_module(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))
    freqs_complex = rope.precompute_theta_pos_freqs(d_model // n_heads, seq_len * 2)[
        :seq_len
    ]

    # Test with RoPE
    output_rope, _ = gqa_module(x, freqs_complex=freqs_complex)
    assert not jnp.any(jnp.isnan(output_rope))
    assert not jnp.any(jnp.isinf(output_rope))

    # Test without RoPE
    output_no_rope, _ = gqa_module(x)
    assert not jnp.any(jnp.isnan(output_no_rope))
    assert not jnp.any(jnp.isinf(output_no_rope))


def test_sliding_window_attention():
    """Test GroupedQueryAttention with sliding window."""
    d_model = 128
    num_heads = 8
    num_kv_heads = 4  # Using half the number of heads for key-value
    batch_size = 2
    seq_length = 10
    window_size = seq_length

    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_length, d_model))
    freqs_complex = rope.precompute_theta_pos_freqs(
        d_model // num_heads, seq_length * 2
    )[:seq_length]

    attention_layer = GroupedQueryAttention(
        d_model,
        num_heads,
        n_kv_heads=num_kv_heads,
        sliding_window_size=window_size,
        rngs=nnx.Rngs(0),
    )

    # Test with RoPE
    output_rope, _ = attention_layer(x, freqs_complex=freqs_complex)
    assert output_rope.shape == (batch_size, seq_length, d_model)

    # Test without RoPE
    output_no_rope, _ = attention_layer(x)
    assert output_no_rope.shape == (batch_size, seq_length, d_model)

    sliding_window_attention_layer = GroupedQueryAttention(
        d_model,
        num_heads,
        n_kv_heads=num_kv_heads,
        sliding_window_size=window_size,
        rngs=nnx.Rngs(0),
    )

    # Test with RoPE
    sliding_output_rope, _ = sliding_window_attention_layer(
        x, freqs_complex=freqs_complex
    )
    assert sliding_output_rope.shape == (batch_size, seq_length, d_model)
    assert jnp.allclose(output_rope, sliding_output_rope, atol=1e-5)

    # Test without RoPE
    sliding_output_no_rope, _ = sliding_window_attention_layer(x)
    assert sliding_output_no_rope.shape == (batch_size, seq_length, d_model)
    assert jnp.allclose(output_no_rope, sliding_output_no_rope, atol=1e-5)


def test_grouped_query_attention_sliding_window_consistency():
    """Test GroupedQueryAttention output is consistent with/without sliding window."""
    d_model = 128
    num_heads = 8
    num_kv_heads = 4  # Using half the number of heads for key-value
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
    attention_layer_without_window = GroupedQueryAttention(
        d_model, num_heads, n_kv_heads=num_kv_heads, rngs=nnx.Rngs(0)
    )

    # Test with RoPE
    output_without_window_rope, _ = attention_layer_without_window(
        x, freqs_complex=freqs_complex
    )

    # Test without RoPE
    output_without_window_no_rope, _ = attention_layer_without_window(x)

    # Attention layer with sliding window
    attention_layer_with_window = GroupedQueryAttention(
        d_model,
        num_heads,
        n_kv_heads=num_kv_heads,
        sliding_window_size=window_size,
        rngs=nnx.Rngs(0),
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
