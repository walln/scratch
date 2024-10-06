"""Tests for the grouped query attention module."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from scratch.deep_learning.layers.attention.grouped_query_attention import (
    GroupedQueryAttention,
)


def create_gqa_module(
    d_model, n_heads, n_kv_heads, max_batch_size, max_seq_len, use_kv_cache=True
):
    """Create a grouped query attention module."""
    return GroupedQueryAttention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        use_kv_cache=use_kv_cache,
        rngs=nnx.Rngs(0),
    )


def test_output_shape():
    """Test that the output shape of the grouped query attention module is correct."""
    d_model, n_heads, n_kv_heads, max_batch_size, max_seq_len = 512, 8, 2, 4, 32
    batch_size, seq_len = max_batch_size // 2, max_seq_len // 2

    gqa_module = create_gqa_module(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))
    freqs_complex = GroupedQueryAttention.precompute_theta_pos_freqs(
        d_model // n_heads, seq_len * 2
    )[:seq_len]

    output = gqa_module(x, freqs_complex=freqs_complex)
    assert output.shape == (batch_size, seq_len, d_model)


def test_kv_cache_update():
    """Test that the KV cache is updated correctly."""
    d_model, n_heads, n_kv_heads, max_batch_size, max_seq_len = 512, 8, 2, 4, 32
    batch_size, seq_len = max_batch_size // 2, max_seq_len // 2

    gqa_module = create_gqa_module(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))
    freqs_complex = GroupedQueryAttention.precompute_theta_pos_freqs(
        d_model // n_heads, seq_len * 2
    )[:seq_len]

    gqa_module(x, freqs_complex=freqs_complex, start_pos=0)
    assert jnp.any(gqa_module.cache_k[:batch_size, :seq_len] != 0)
    assert jnp.any(gqa_module.cache_v[:batch_size, :seq_len] != 0)


def test_no_kv_cache():
    """Test that the KV cache is not used when use_kv_cache is False."""
    d_model, n_heads, n_kv_heads, max_batch_size, max_seq_len = 512, 8, 2, 4, 32
    gqa_module = create_gqa_module(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        use_kv_cache=False,
    )

    assert not hasattr(gqa_module, "cache_k")
    assert not hasattr(gqa_module, "cache_v")


@pytest.mark.parametrize(
    ("d_model", "n_heads", "n_kv_heads"),
    [(512, 8, 2), (512, 8, 4), (768, 12, 3), (1024, 16, 4)],
)
def test_different_head_configurations(d_model: int, n_heads: int, n_kv_heads: int):
    """Test that the module works with different head configurations."""
    max_batch_size, max_seq_len = 4, 32
    batch_size, seq_len = max_batch_size // 2, max_seq_len // 2

    gqa_module = create_gqa_module(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))
    freqs_complex = GroupedQueryAttention.precompute_theta_pos_freqs(
        d_model // n_heads, seq_len * 2
    )[0 : 0 + seq_len]

    output = gqa_module(x, freqs_complex=freqs_complex)
    assert output.shape == (batch_size, seq_len, d_model)


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
            max_batch_size=4,
            max_seq_len=32,
        )


def test_forward_backward():
    """Test the forward and backward pass of the grouped query attention module."""
    d_model, n_heads, n_kv_heads, max_batch_size, max_seq_len = 512, 8, 2, 4, 32
    batch_size, seq_len = max_batch_size // 2, max_seq_len // 2

    gqa_module = create_gqa_module(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))
    freqs_complex = GroupedQueryAttention.precompute_theta_pos_freqs(
        d_model // n_heads, seq_len * 2
    )[:seq_len]

    def loss_fn(model):
        output = model(x, freqs_complex=freqs_complex)
        return jnp.mean(output**2)

    loss, grads = nnx.value_and_grad(loss_fn)(gqa_module)

    # Remove the KV cache from the params for the shape check. The NNX lifted transform
    # for computing gradients ignores the KV cache by default, but they are still
    # present in the parameter pytree.
    graphdef, params, caches = nnx.split(gqa_module, nnx.Param, jnp.ndarray)
    shape_check = jax.tree_util.tree_map(lambda p, g: p.shape == g.shape, params, grads)
    assert jax.tree_util.tree_all(shape_check), f"Shapes don't match: {shape_check}"


@pytest.mark.parametrize("start_pos", [0, 8, 16])
def test_incremental_forward(start_pos):
    """Test that the grouped query attention module can be used incrementally.

    This test checks that the module can be used to compute the output for a
    sequence of positions that are not consecutive.
    """
    d_model, n_heads, n_kv_heads, max_batch_size, max_seq_len = 512, 8, 2, 4, 32
    batch_size, seq_len = max_batch_size // 2, max_seq_len // 2

    gqa_module = create_gqa_module(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))
    freqs_complex = GroupedQueryAttention.precompute_theta_pos_freqs(
        d_model // n_heads, (start_pos + seq_len) * 2
    )[start_pos : start_pos + seq_len]

    output = gqa_module(x, freqs_complex=freqs_complex, start_pos=start_pos)
    assert output.shape == (batch_size, seq_len, d_model)
    assert jnp.any(
        gqa_module.cache_k[:batch_size, start_pos : start_pos + seq_len] != 0
    )


def test_numerical_stability():
    """Test that the grouped query attention module is numerically stable."""
    d_model, n_heads, n_kv_heads, max_batch_size, max_seq_len = 512, 8, 2, 4, 32
    batch_size, seq_len = max_batch_size // 2, max_seq_len // 2

    gqa_module = create_gqa_module(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))
    freqs_complex = GroupedQueryAttention.precompute_theta_pos_freqs(
        d_model // n_heads, seq_len * 2
    )[:seq_len]

    output = gqa_module(x, freqs_complex=freqs_complex)
    assert not jnp.any(jnp.isnan(output))
    assert not jnp.any(jnp.isinf(output))
