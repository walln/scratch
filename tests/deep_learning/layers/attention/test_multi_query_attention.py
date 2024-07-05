"""Tests for MultiQueryAttention layer."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from scratch.deep_learning.layers.attention.multi_query_attention import (
    MultiQueryAttention,
)


class DummyModel(nnx.Module):
    """Dummy model that uses MultiQueryAttention."""

    def __init__(self, embed_dim: int, num_heads: int, *, rngs: nnx.Rngs):
        """Initialize the model.

        Args:
            embed_dim: The embedding dimension.
            num_heads: The number of attention heads.
            rngs: The random number generators.
        """
        self.mqa = MultiQueryAttention(embed_dim, num_heads, rngs=rngs)

    def __call__(
        self, x: jnp.ndarray, mask: jnp.ndarray | None = None, deterministic=True
    ):
        """Forward pass of the model.

        Args:
            x: The input tensor.
            mask: The mask tensor.
            deterministic: Whether to use deterministic behavior.

        Returns:
            The output tensor.
        """
        return self.mqa(x, x, x, mask=mask, deterministic=deterministic)


@pytest.fixture()
def setup_model():
    """Set up the model and input tensors and mask."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (8, 128, 64))  # (batch_size, sequence_length, model_dim)
    mask = jax.random.bernoulli(key, 0.1, (8, 128))
    model = DummyModel(embed_dim=64, num_heads=8, rngs=nnx.Rngs(0))
    return model, x, mask


def test_output_shape(setup_model):
    """Test the output shape of the model."""
    model, x, mask = setup_model
    output = model(x, mask=mask)
    assert output.shape == (8, 128, 64)


def test_output_type(setup_model):
    """Test the output type of the model."""
    model, x, mask = setup_model
    output = model(x, mask=mask)
    assert isinstance(output, jnp.ndarray)


def test_mask_application(setup_model):
    """Test the application of the mask."""
    model, x, mask = setup_model
    output_with_mask = model(x, mask=mask)
    output_without_mask = model(x)
    assert not jnp.allclose(output_with_mask, output_without_mask)


def test_deterministic_behavior(setup_model):
    """Test the deterministic behavior of the attention dropout."""
    model, x, mask = setup_model
    deterministic_output1 = model(x, mask=mask, deterministic=True)
    deterministic_output2 = model(x, mask=mask, deterministic=True)
    assert jnp.allclose(deterministic_output1, deterministic_output2)


def test_non_deterministic_behavior(setup_model):
    """Test the non-deterministic behavior of the attention dropout."""
    model, x, mask = setup_model
    nondeterministic_output1 = model(x, mask=mask, deterministic=False)
    nondeterministic_output2 = model(x, mask=mask, deterministic=False)
    assert not jnp.allclose(nondeterministic_output1, nondeterministic_output2)
