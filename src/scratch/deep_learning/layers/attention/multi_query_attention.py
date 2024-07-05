"""Multi-Query Attention (MQA).

This module implements Multi-Query Attention as described in "Fast Transformer
Decoding: One Write-Head is All You Need" by Noam Shazeer. The main idea is to use
a single set of keys and values across all attention heads, improving efficiency
while maintaining performance.

Reference:
    Shazeer, Noam. "Fast Transformer Decoding: One Write-Head is All You Need."
    arXiv preprint arXiv:1911.02150 (2019). https://arxiv.org/abs/1911.02150
"""

import jax.numpy as jnp
from flax import nnx


class MultiQueryAttention(nnx.Module):
    """Multi-Query Attention (MQA).

    This module performs multi-query attention, where the keys and values are shared
    across all attention heads.
    """

    def __init__(
        self, embed_dim: int, n_heads: int, dropout_rate=0.1, *, rngs: nnx.Rngs
    ):
        """Initializes the MultiQueryAttention module.

        Args:
            embed_dim: The dimension of the input embeddings.
            n_heads: The number of attention heads.
            dropout_rate: The dropout rate for regularization.
            rngs: Random number generators for initializing parameters.
        """
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.embed_dim = embed_dim

        assert (
            self.head_dim * n_heads == embed_dim
        ), "embed_dim must be divisible by n_heads"

        self.q_proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.k_proj = nnx.Linear(embed_dim, self.head_dim, rngs=rngs)
        self.v_proj = nnx.Linear(embed_dim, self.head_dim, rngs=rngs)
        self.out_proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)

        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, query, key, value, mask=None, deterministic=True):
        """Applies the multi-query attention mechanism.

        Args:
            query: Query tensor of shape (batch_size, seq_length, embed_dim).
            key: Key tensor of shape (batch_size, seq_length, embed_dim).
            value: Value tensor of shape (batch_size, seq_length, embed_dim).
            mask: Attention mask of shape (batch_size, seq_length). Defaults to None.
            deterministic: If True, applies deterministic operations (e.g., no dropout).
                           Defaults to True.

        Returns:
            Output tensor of shape (batch_size, seq_length, embed_dim).
        """
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.reshape(query.shape[0], query.shape[1], self.n_heads, self.head_dim)
        k = k.reshape(query.shape[0], -1, self.head_dim)  # Shared across heads
        v = v.reshape(query.shape[0], -1, self.head_dim)  # Shared across heads

        attn_weights = jnp.einsum("...qhd,...kd->...hqk", q, k) / jnp.sqrt(
            self.head_dim
        )
        if mask is not None:
            attn_weights = jnp.where(mask[:, None, None, :], attn_weights, -1e9)

        attn_weights = nnx.softmax(attn_weights, axis=-1)
        attn_weights = self.dropout(attn_weights, deterministic=deterministic)

        attn_output = jnp.einsum("...hqk,...kd->...qhd", attn_weights, v)
        attn_output = attn_output.reshape(query.shape[0], query.shape[1], -1)

        return self.out_proj(attn_output)
