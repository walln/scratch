"""Tests for MultiQueryAttention layer."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from scratch.deep_learning.layers.attention import rope
from scratch.deep_learning.layers.attention.kv_cache import LayerKVCache
from scratch.deep_learning.layers.attention.multi_query_attention import (
    MultiQueryAttention,
)


def create_mqa_module(
    d_model,
    n_heads,
    dropout_rate=0.1,
):
    """Create a multi-query attention module."""
    return MultiQueryAttention(
        d_model=d_model,
        n_heads=n_heads,
        dropout_rate=dropout_rate,
        rngs=nnx.Rngs(0),
    )


def test_output_shape():
    """Test that the output shape of the multi-query attention module is correct."""
    d_model, n_heads = 512, 8
    batch_size, seq_len = 4, 32

    mqa_module = create_mqa_module(d_model=d_model, n_heads=n_heads)
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))
    freqs_complex = rope.precompute_theta_pos_freqs(d_model // n_heads, seq_len * 2)[
        :seq_len
    ]

    # Test with RoPE
    output_rope, _ = mqa_module(x, freqs_complex=freqs_complex)
    assert output_rope.shape == (batch_size, seq_len, d_model)

    # Test without RoPE
    output_no_rope, _ = mqa_module(x)
    assert output_no_rope.shape == (batch_size, seq_len, d_model)


def test_kv_cache_update():
    """Test that the KV cache is updated correctly."""
    d_model, n_heads = 512, 8
    batch_size, seq_len = 4, 32

    mqa_module = create_mqa_module(d_model=d_model, n_heads=n_heads)
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))
    kv_cache = LayerKVCache.create(batch_size, seq_len, 1, d_model // n_heads)
    freqs_complex = rope.precompute_theta_pos_freqs(d_model // n_heads, seq_len * 2)[
        :seq_len
    ]

    # Test with RoPE
    _, new_kv_cache_rope = mqa_module(
        x, start_pos=0, kv_cache=kv_cache, freqs_complex=freqs_complex
    )
    assert new_kv_cache_rope is not None
    assert jnp.any(new_kv_cache_rope.k[:batch_size, :seq_len] != 0)
    assert jnp.any(new_kv_cache_rope.v[:batch_size, :seq_len] != 0)

    # Test without RoPE
    _, new_kv_cache_no_rope = mqa_module(x, start_pos=0, kv_cache=kv_cache)
    assert new_kv_cache_no_rope is not None
    assert jnp.any(new_kv_cache_no_rope.k[:batch_size, :seq_len] != 0)
    assert jnp.any(new_kv_cache_no_rope.v[:batch_size, :seq_len] != 0)


def test_no_kv_cache():
    """Test that the KV cache is not used when not provided."""
    d_model, n_heads = 512, 8
    batch_size, seq_len = 4, 32

    mqa_module = create_mqa_module(d_model=d_model, n_heads=n_heads)
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))
    freqs_complex = rope.precompute_theta_pos_freqs(d_model // n_heads, seq_len * 2)[
        :seq_len
    ]

    # Test with RoPE
    output_rope, new_kv_cache_rope = mqa_module(x, freqs_complex=freqs_complex)
    assert new_kv_cache_rope is None

    # Test without RoPE
    output_no_rope, new_kv_cache_no_rope = mqa_module(x)
    assert new_kv_cache_no_rope is None


@pytest.mark.parametrize(
    ("d_model", "n_heads"),
    [(512, 8), (768, 12), (1024, 16)],
)
def test_different_head_configurations(d_model: int, n_heads: int):
    """Test that the module works with different head configurations."""
    batch_size, seq_len = 4, 32

    mqa_module = create_mqa_module(d_model=d_model, n_heads=n_heads)
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))
    freqs_complex = rope.precompute_theta_pos_freqs(d_model // n_heads, seq_len * 2)[
        :seq_len
    ]

    # Test with RoPE
    output_rope, _ = mqa_module(x, freqs_complex=freqs_complex)
    assert output_rope.shape == (batch_size, seq_len, d_model)

    # Test without RoPE
    output_no_rope, _ = mqa_module(x)
    assert output_no_rope.shape == (batch_size, seq_len, d_model)


@pytest.mark.parametrize(
    ("d_model", "n_heads"),
    [
        (512, 7),  # d_model not divisible by n_heads
        (768, 13),  # d_model not divisible by n_heads
    ],
)
def test_invalid_head_configuration(d_model: int, n_heads: int):
    """Test that an assertion error is raised when the head configuration is invalid."""
    with pytest.raises(AssertionError):
        create_mqa_module(d_model=d_model, n_heads=n_heads)


def test_forward_backward():
    """Test the forward and backward pass of the multi-query attention module."""
    d_model, n_heads = 512, 8
    batch_size, seq_len = 4, 32

    mqa_module = create_mqa_module(d_model=d_model, n_heads=n_heads)
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))
    freqs_complex = rope.precompute_theta_pos_freqs(d_model // n_heads, seq_len * 2)[
        :seq_len
    ]

    def loss_fn(model):
        output, _ = model(x, freqs_complex=freqs_complex)
        return jnp.mean(output**2)

    loss, grads = nnx.value_and_grad(loss_fn)(mqa_module)

    graphdef, params, others = nnx.split(mqa_module, nnx.Param, ...)

    shape_check = jax.tree_util.tree_map(lambda p, g: p.shape == g.shape, params, grads)
    assert jax.tree_util.tree_all(shape_check), f"Shapes don't match: {shape_check}"


@pytest.mark.parametrize("start_pos", [0, 8, 16])
def test_incremental_forward(start_pos):
    """Test that the multi-query attention module can be used incrementally."""
    d_model, n_heads = 512, 8
    batch_size, seq_len = 4, 32

    mqa_module = create_mqa_module(d_model=d_model, n_heads=n_heads)
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))
    kv_cache = LayerKVCache.create(batch_size, seq_len, 1, d_model // n_heads)
    freqs_complex = rope.precompute_theta_pos_freqs(d_model // n_heads, seq_len * 2)[
        :seq_len
    ]

    # Test with RoPE
    output_rope, new_kv_cache_rope = mqa_module(
        x, start_pos=start_pos, kv_cache=kv_cache, freqs_complex=freqs_complex
    )
    assert output_rope.shape == (batch_size, seq_len, d_model)
    assert new_kv_cache_rope is not None
    assert jnp.any(new_kv_cache_rope.k[:, start_pos : start_pos + seq_len, :, :] != 0)

    # Test without RoPE
    output_no_rope, new_kv_cache_no_rope = mqa_module(
        x, start_pos=start_pos, kv_cache=kv_cache
    )
    assert output_no_rope.shape == (batch_size, seq_len, d_model)
    assert new_kv_cache_no_rope is not None
    assert jnp.any(
        new_kv_cache_no_rope.k[:, start_pos : start_pos + seq_len, :, :] != 0
    )


def test_numerical_stability():
    """Test that the multi-query attention module is numerically stable."""
    d_model, n_heads = 512, 8
    batch_size, seq_len = 4, 32

    mqa_module = create_mqa_module(d_model=d_model, n_heads=n_heads)
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))
    freqs_complex = rope.precompute_theta_pos_freqs(d_model // n_heads, seq_len * 2)[
        :seq_len
    ]

    # Test with RoPE
    output_rope, _ = mqa_module(x, freqs_complex=freqs_complex)
    assert not jnp.any(jnp.isnan(output_rope))
    assert not jnp.any(jnp.isinf(output_rope))

    # Test without RoPE
    output_no_rope, _ = mqa_module(x)
    assert not jnp.any(jnp.isnan(output_no_rope))
    assert not jnp.any(jnp.isinf(output_no_rope))


def test_masking():
    """Test that the masking in multi-query attention works correctly."""
    d_model, n_heads = 512, 8
    batch_size, seq_len = 4, 32

    mqa_module = create_mqa_module(d_model=d_model, n_heads=n_heads)
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))
    freqs_complex = rope.precompute_theta_pos_freqs(d_model // n_heads, seq_len * 2)[
        :seq_len
    ]

    # Create a simple mask where the second half of the sequence is masked
    mask = jnp.ones((batch_size, seq_len))
    mask = mask.at[:, seq_len // 2 :].set(0)

    # Test with RoPE
    output_with_mask_rope, _ = mqa_module(x, mask=mask, freqs_complex=freqs_complex)
    output_without_mask_rope, _ = mqa_module(x, freqs_complex=freqs_complex)

    # The outputs should be different when a mask is applied
    assert not jnp.allclose(output_with_mask_rope, output_without_mask_rope)

    # Test without RoPE
    output_with_mask_no_rope, _ = mqa_module(x, mask=mask)
    output_without_mask_no_rope, _ = mqa_module(x)

    # The outputs should be different when a mask is applied
    assert not jnp.allclose(output_with_mask_no_rope, output_without_mask_no_rope)


def test_deterministic_mode():
    """Test that the deterministic mode in multi-query attention works correctly."""
    d_model, n_heads = 512, 8
    batch_size, seq_len = 4, 32

    mqa_module = create_mqa_module(
        d_model=d_model,
        n_heads=n_heads,
        dropout_rate=0.5,  # Set a high dropout rate to make the effect more noticeable
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))
    freqs_complex = rope.precompute_theta_pos_freqs(d_model // n_heads, seq_len * 2)[
        :seq_len
    ]

    # Run the module multiple times in deterministic mode with RoPE
    deterministic_outputs_rope = [
        mqa_module(x, deterministic=True, freqs_complex=freqs_complex)[0]
        for _ in range(5)
    ]

    # All outputs should be the same in deterministic mode
    for output in deterministic_outputs_rope[1:]:
        assert jnp.allclose(output, deterministic_outputs_rope[0])

    # Run the module multiple times in non-deterministic mode with RoPE
    non_deterministic_outputs_rope = [
        mqa_module(x, deterministic=False, freqs_complex=freqs_complex)[0]
        for _ in range(5)
    ]

    # Outputs should be different in non-deterministic mode (with high probability)
    assert not all(
        jnp.allclose(output, non_deterministic_outputs_rope[0])
        for output in non_deterministic_outputs_rope[1:]
    )

    # Run the module multiple times in deterministic mode without RoPE
    deterministic_outputs_no_rope = [
        mqa_module(x, deterministic=True)[0] for _ in range(5)
    ]

    # All outputs should be the same in deterministic mode
    for output in deterministic_outputs_no_rope[1:]:
        assert jnp.allclose(output, deterministic_outputs_no_rope[0])

    # Run the module multiple times in non-deterministic mode without RoPE
    non_deterministic_outputs_no_rope = [
        mqa_module(x, deterministic=False)[0] for _ in range(5)
    ]

    # Outputs should be different in non-deterministic mode (with high probability)
    assert not all(
        jnp.allclose(output, non_deterministic_outputs_no_rope[0])
        for output in non_deterministic_outputs_no_rope[1:]
    )


def test_sliding_window_attention():
    """Test MultiQueryAttention with sliding window."""
    d_model = 128
    num_heads = 8
    batch_size = 2
    seq_length = 10
    window_size = seq_length

    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_length, d_model))
    freqs_complex = rope.precompute_theta_pos_freqs(
        d_model // num_heads, seq_length * 2
    )[:seq_length]

    attention_layer = MultiQueryAttention(
        d_model, num_heads, sliding_window_size=window_size, rngs=nnx.Rngs(0)
    )

    # Test with RoPE
    output_rope, _ = attention_layer(x, freqs_complex=freqs_complex)
    assert output_rope.shape == (batch_size, seq_length, d_model)

    # Test without RoPE
    output_no_rope, _ = attention_layer(x)
    assert output_no_rope.shape == (batch_size, seq_length, d_model)

    sliding_window_attention_layer = MultiQueryAttention(
        d_model, num_heads, sliding_window_size=window_size, rngs=nnx.Rngs(0)
    )

    # Test with RoPE
    sliding_output_rope, _ = sliding_window_attention_layer(
        x, freqs_complex=freqs_complex
    )
    assert sliding_output_rope.shape == (batch_size, seq_length, d_model)

    # Test without RoPE
    sliding_output_no_rope, _ = sliding_window_attention_layer(x)
    assert sliding_output_no_rope.shape == (batch_size, seq_length, d_model)

    assert jnp.allclose(output_rope, sliding_output_rope, atol=1e-5)
    assert jnp.allclose(output_no_rope, sliding_output_no_rope, atol=1e-5)


def test_multi_query_attention_sliding_window_consistency():
    """Test if MultiQueryAttention output is consistent with/without sliding window."""
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
    attention_layer_without_window = MultiQueryAttention(
        d_model, num_heads, rngs=nnx.Rngs(0)
    )

    # Test with RoPE
    output_without_window_rope, _ = attention_layer_without_window(
        x, freqs_complex=freqs_complex
    )

    # Test without RoPE
    output_without_window_no_rope, _ = attention_layer_without_window(x)

    # Attention layer with sliding window
    attention_layer_with_window = MultiQueryAttention(
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
