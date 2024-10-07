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

from scratch.deep_learning.layers.attention.kv_cache import LayerKVCache


# TODO(walln): Add support for RoPE
class MultiQueryAttention(nnx.Module):
    """Multi-Query Attention (MQA).

    This module performs multi-query attention, where the keys and values are shared
    across all attention heads.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout_rate=0.1,
        *,
        rngs: nnx.Rngs,
    ):
        """Initializes the MultiQueryAttention module.

        Args:
            d_model: The dimension of the input embeddings.
            n_heads: The number of attention heads.
            dropout_rate: The dropout rate for regularization.
            rngs: Random number generators for initializing parameters.
        """
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.q_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.k_proj = nnx.Linear(d_model, self.head_dim, rngs=rngs)
        self.v_proj = nnx.Linear(d_model, self.head_dim, rngs=rngs)
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)

        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(
        self,
        x,
        mask=None,
        deterministic=True,
        start_pos=0,
        kv_cache: LayerKVCache | None = None,
    ):
        """Applies the multi-query attention mechanism.

        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model).
            mask: Attention mask of shape (batch_size, seq_length). Defaults to None.
            deterministic: If True, applies deterministic operations (e.g., no dropout).
                Defaults to True.
            start_pos: The start position of the input sequence. Used only if
                `use_kv_cache` is True.
            kv_cache: External KVCache to use for caching keys and values.
                Defaults to None.

        Returns:
            Tuple of tensor (batch_size, seq_length, d_model) and updated KVCache.
        """
        q = self.q_proj(x)  # Shape (batch_size, seq_length, d_model)
        k = self.k_proj(x)  # Shape (batch_size, seq_length, head_dim)
        v = self.v_proj(x)  # Shape (batch_size, seq_length, head_dim)

        # Reshape q to include n_heads
        q = q.reshape(x.shape[0], x.shape[1], self.n_heads, self.head_dim)  # (bqhd)

        # Ensure k and v always have a singleton n_heads dimension
        k = k.reshape(x.shape[0], x.shape[1], 1, self.head_dim)  # (bq1d)
        v = v.reshape(x.shape[0], x.shape[1], 1, self.head_dim)  # (bq1d)

        updated_kv_cache = None
        if kv_cache is not None:
            k, v, updated_kv_cache = kv_cache.update(
                xk=k,
                xv=v,
                cur_pos=start_pos,
                n_rep=self.n_heads,
            )

        # raise ValueError(f"q: {q.shape}, k: {k.shape}, v: {v.shape}")
        attn_weights = jnp.einsum("bqhd, bkhd -> bhqk", q, k) / jnp.sqrt(self.head_dim)
        if mask is not None:
            attn_weights = jnp.where(mask[:, None, None, :], attn_weights, -1e9)

        attn_weights = nnx.softmax(attn_weights, axis=-1)
        attn_weights = self.dropout(attn_weights, deterministic=deterministic)

        attn_output = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, v)
        attn_output = attn_output.reshape(x.shape[0], x.shape[1], -1)

        output = self.out_proj(attn_output)

        return output, updated_kv_cache
