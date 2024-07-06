"""Dot-product attention layer.

Computes the dot products of the query with all values and
applies a softmax function to get the weights. The attention weights
are then used to compute the context tensor.

Based on the dot-product attention mechanism proposed in [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
"""

import jax
import jax.numpy as jnp
from flax import nnx


class DotProductAttention(nnx.Module):
    """Dot-product attention layer."""

    def __call__(
        self, query: jnp.ndarray, value: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute dot-product attention forward pass.

        Args:
            query: The query tensor.
            value: The value tensor.

        Returns:
            The context tensor and the attention tensor.
        """
        scores = jnp.matmul(query, value.transpose((0, 2, 1)))
        attention = jax.nn.softmax(scores, axis=-1)
        context = jnp.matmul(attention, value)

        return context, attention
