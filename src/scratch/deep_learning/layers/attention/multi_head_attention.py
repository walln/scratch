"""Multi-head attention layer.

Applies attention mechanisms to multiple query, key, and value inputs in parallel.
Each attention head computes scaled dot-product attention independently. The results
are concatenated and linearly transformed to produce the final output. This allows
the model to jointly attend to information from different representation subspaces
at different positions.

Originally proposed in [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
"""

import jax.numpy as jnp
from flax import nnx
from scratch.deep_learning.layers.attention.scaled_dot_product_attention import (
    ScaledDotProductAttention,
)


class MultiHeadAttention(nnx.Module):
    """Multi-head attention layer."""

    def __init__(self, d_model: int, num_heads: int, *, rngs: nnx.Rngs):
        """Initialize MultiHeadAttention.

        Args:
            d_model: The dimension of the model.
            num_heads: The number of attention heads.
            rngs: The random number generators.
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Linear layers for query, key, value, and output
        self.q_linear = nnx.Linear(d_model, d_model, rngs=rngs)
        self.k_linear = nnx.Linear(d_model, d_model, rngs=rngs)
        self.v_linear = nnx.Linear(d_model, d_model, rngs=rngs)
        self.out_linear = nnx.Linear(d_model, d_model, rngs=rngs)

        # Scaled dot-product attention layer
        self.attention = ScaledDotProductAttention(self.d_head)

    def split_heads(self, x: jnp.ndarray) -> jnp.ndarray:
        """Split the last dimension into (num_heads, d_head).

        Args:
            x: The input tensor of shape (batch_size, seq_length, d_model).

        Returns:
            The tensor of shape (batch_size, num_heads, seq_length, d_head).
        """
        batch_size, seq_length, _ = x.shape
        x = x.reshape(batch_size, seq_length, self.num_heads, self.d_head)
        return x.transpose((0, 2, 1, 3))

    def combine_heads(self, x: jnp.ndarray) -> jnp.ndarray:
        """Combine the (num_heads, d_head) dimensions back into (d_model).

        Args:
            x: The input tensor of shape (batch_size, num_heads, seq_length, d_head).

        Returns:
            The tensor of shape (batch_size, seq_length, d_model).
        """
        batch_size, num_heads, seq_length, d_head = x.shape
        x = x.transpose((0, 2, 1, 3)).reshape(batch_size, seq_length, self.d_model)
        return x

    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute multi-head attention forward pass.

        Args:
            query: The query tensor.
            key: The key tensor.
            value: The value tensor.
            mask: The mask tensor. Defaults to None.

        Returns:
            The output tensor and the attention tensor.
        """
        batch_size = query.shape[0]
        seq_length = query.shape[1]

        # Linear projections
        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)

        # Split heads
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        # Reshape mask to match the shape of the scores
        if mask is not None:
            mask = mask[:, None, :, :].repeat(self.num_heads, axis=1)
            mask = mask.reshape(batch_size * self.num_heads, seq_length, seq_length)

        # Reshape query, key, value for attention calculation
        query = query.reshape(batch_size * self.num_heads, seq_length, self.d_head)
        key = key.reshape(batch_size * self.num_heads, seq_length, self.d_head)
        value = value.reshape(batch_size * self.num_heads, seq_length, self.d_head)

        # Scaled dot-product attention
        context, attention = self.attention(query, key, value, mask)

        # Combine heads
        context = context.reshape(batch_size, self.num_heads, seq_length, self.d_head)
        context = self.combine_heads(context)

        # Final linear projection
        output = self.out_linear(context)

        return output, attention
