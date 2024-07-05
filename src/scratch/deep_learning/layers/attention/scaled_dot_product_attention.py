"""Scaled dot-product attention layer.

Computes the dot products of the query with all keys, scales the dot products
by a factor of square root of the key dimension, and applies a softmax function.
Originally proposed in [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
"""

import jax
import jax.numpy as jnp
from flax import nnx


class ScaledDotProductAttention(nnx.Module):
    """Scaled dot-product attention layer."""

    def __init__(self, d_head: int):
        """Initialize ScaledDotProductAttention.

        Args:
            d_head: The head dimension.
        """
        self.d_head = d_head

    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute scaled dot-product attention forward pass.

        Args:
            query: The query tensor.
            key: The key tensor.
            value: The value tensor.
            mask: The mask tensor. Defaults to None.

        Returns:
            The context tensor and the attention tensor.
        """
        d_k = key.shape[-1]

        scores = jnp.matmul(query, key.transpose((0, 2, 1))) / jnp.sqrt(d_k)

        if mask is not None:
            scores = jnp.where(mask, -jnp.inf, scores)

        # Apply softmax to get attention weights
        attention = jax.nn.softmax(scores, axis=-1)

        # Compute the context tensor
        context = jnp.matmul(attention, value)

        return context, attention
