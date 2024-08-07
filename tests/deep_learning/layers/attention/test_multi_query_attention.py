"""Tests for MultiQueryAttention layer."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from scratch.deep_learning.layers.attention.multi_query_attention import (
    MultiQueryAttention,
)


def create_mqa_module(
    d_model,
    n_heads,
    dropout_rate=0.1,
    max_batch_size=32,
    max_seq_len=2048,
    use_kv_cache=True,
):
    """Create a multi-query attention module."""
    return MultiQueryAttention(
        d_model=d_model,
        n_heads=n_heads,
        dropout_rate=dropout_rate,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        use_kv_cache=use_kv_cache,
        rngs=nnx.Rngs(0),
    )


def test_output_shape():
    """Test that the output shape of the multi-query attention module is correct."""
    d_model, n_heads, max_batch_size, max_seq_len = 512, 8, 4, 32
    batch_size, seq_len = max_batch_size // 2, max_seq_len // 2

    mqa_module = create_mqa_module(
        d_model=d_model,
        n_heads=n_heads,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))

    output = mqa_module(x)
    assert output.shape == (batch_size, seq_len, d_model)


def test_kv_cache_update():
    """Test that the KV cache is updated correctly."""
    d_model, n_heads, max_batch_size, max_seq_len = 512, 8, 4, 32
    batch_size, seq_len = max_batch_size // 2, max_seq_len // 2

    mqa_module = create_mqa_module(
        d_model=d_model,
        n_heads=n_heads,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))

    mqa_module(x, start_pos=0)
    assert jnp.any(mqa_module.cache_k[:batch_size, :seq_len] != 0)
    assert jnp.any(mqa_module.cache_v[:batch_size, :seq_len] != 0)


def test_no_kv_cache():
    """Test that the KV cache is not used when use_kv_cache is False."""
    d_model, n_heads, max_batch_size, max_seq_len = 512, 8, 4, 32
    mqa_module = create_mqa_module(
        d_model=d_model,
        n_heads=n_heads,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        use_kv_cache=False,
    )

    assert not hasattr(mqa_module, "cache_k")
    assert not hasattr(mqa_module, "cache_v")


@pytest.mark.parametrize(
    ("d_model", "n_heads"),
    [(512, 8), (768, 12), (1024, 16)],
)
def test_different_head_configurations(d_model: int, n_heads: int):
    """Test that the module works with different head configurations."""
    max_batch_size, max_seq_len = 4, 32
    batch_size, seq_len = max_batch_size // 2, max_seq_len // 2

    mqa_module = create_mqa_module(
        d_model=d_model,
        n_heads=n_heads,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))

    output = mqa_module(x)
    assert output.shape == (batch_size, seq_len, d_model)


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
        create_mqa_module(
            d_model=d_model,
            n_heads=n_heads,
            max_batch_size=4,
            max_seq_len=32,
        )


def test_forward_backward():
    """Test the forward and backward pass of the multi-query attention module."""
    d_model, n_heads, max_batch_size, max_seq_len = 512, 8, 4, 32
    batch_size, seq_len = max_batch_size // 2, max_seq_len // 2

    mqa_module = create_mqa_module(
        d_model=d_model,
        n_heads=n_heads,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))

    def loss_fn(model):
        output = model(x)
        return jnp.mean(output**2)

    loss, grads = nnx.value_and_grad(loss_fn)(mqa_module)

    graphdef, params, others = nnx.split(mqa_module, nnx.Param, ...)

    shape_check = jax.tree_util.tree_map(lambda p, g: p.shape == g.shape, params, grads)
    assert jax.tree_util.tree_all(shape_check), f"Shapes don't match: {shape_check}"


@pytest.mark.parametrize("start_pos", [0, 8, 16])
def test_incremental_forward(start_pos):
    """Test that the multi-query attention module can be used incrementally."""
    d_model, n_heads, max_batch_size, max_seq_len = 512, 8, 4, 32
    batch_size, seq_len = max_batch_size // 2, max_seq_len // 2

    mqa_module = create_mqa_module(
        d_model=d_model,
        n_heads=n_heads,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))

    output = mqa_module(x, start_pos=start_pos)
    assert output.shape == (batch_size, seq_len, d_model)
    assert jnp.any(
        mqa_module.cache_k[:batch_size, start_pos : start_pos + seq_len] != 0
    )


def test_numerical_stability():
    """Test that the multi-query attention module is numerically stable."""
    d_model, n_heads, max_batch_size, max_seq_len = 512, 8, 4, 32
    batch_size, seq_len = max_batch_size // 2, max_seq_len // 2

    mqa_module = create_mqa_module(
        d_model=d_model,
        n_heads=n_heads,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))

    output = mqa_module(x)
    assert not jnp.any(jnp.isnan(output))
    assert not jnp.any(jnp.isinf(output))


def test_masking():
    """Test that the masking in multi-query attention works correctly."""
    d_model, n_heads, max_batch_size, max_seq_len = 512, 8, 4, 32
    batch_size, seq_len = max_batch_size // 2, max_seq_len // 2

    mqa_module = create_mqa_module(
        d_model=d_model,
        n_heads=n_heads,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))

    # Create a simple mask where the second half of the sequence is masked
    mask = jnp.ones((batch_size, seq_len))
    mask = mask.at[:, seq_len // 2 :].set(0)

    output_with_mask = mqa_module(x, mask=mask)
    output_without_mask = mqa_module(x)

    # The outputs should be different when a mask is applied
    assert not jnp.allclose(output_with_mask, output_without_mask)


def test_deterministic_mode():
    """Test that the deterministic mode in multi-query attention works correctly."""
    d_model, n_heads, max_batch_size, max_seq_len = 512, 8, 4, 32
    batch_size, seq_len = max_batch_size // 2, max_seq_len // 2

    mqa_module = create_mqa_module(
        d_model=d_model,
        n_heads=n_heads,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        dropout_rate=0.5,  # Set a high dropout rate to make the effect more noticeable
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))

    # Run the module multiple times in deterministic mode
    deterministic_outputs = [mqa_module(x, deterministic=True) for _ in range(5)]

    # All outputs should be the same in deterministic mode
    for output in deterministic_outputs[1:]:
        assert jnp.allclose(output, deterministic_outputs[0])

    # Run the module multiple times in non-deterministic mode
    non_deterministic_outputs = [mqa_module(x, deterministic=False) for _ in range(5)]

    # Outputs should be different in non-deterministic mode (with high probability)
    assert not all(
        jnp.allclose(output, non_deterministic_outputs[0])
        for output in non_deterministic_outputs[1:]
    )
