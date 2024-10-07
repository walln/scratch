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
from scratch.deep_learning.layers.attention.rope import apply_rotary_emb
from scratch.deep_learning.layers.attention.sliding_window import (
    create_sliding_window_mask,
)


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
        sliding_window_size: int | None = None,
        rngs: nnx.Rngs,
    ):
        """Initializes the MultiQueryAttention module.

        Args:
            d_model: The dimension of the input embeddings.
            n_heads: The number of attention heads.
            dropout_rate: The dropout rate for regularization.
            sliding_window_size: The size of the sliding window for the sliding window
                attention mechanism. Defaults to None.
            rngs: Random number generators for initializing parameters.
        """
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.sliding_window_size = sliding_window_size

        self.q_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.k_proj = nnx.Linear(d_model, self.head_dim, rngs=rngs)
        self.v_proj = nnx.Linear(d_model, self.head_dim, rngs=rngs)
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)

        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        freqs_complex: jnp.ndarray | None = None,
        mask: jnp.ndarray | None = None,
        deterministic: bool = True,
        start_pos: int = 0,
        kv_cache: LayerKVCache | None = None,
    ):
        """Applies the multi-query attention mechanism.

        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model).
            freqs_complex: The frequencies for RoPE embeddings. Defaults to None.
            mask: Attention mask of shape (batch_size, seq_length). Defaults to None.
            deterministic: If True, applies deterministic operations (e.g., no dropout).
                Defaults to True.
            start_pos: The start position of the input sequence. Used only if
                `use_kv_cache` is True.
            kv_cache: External KVCache to use for caching keys and values.
                Defaults to None.

        Returns:
            Tuple of tensor (batch_size, seq_length, d_model) and updated LayerKVCache.
        """
        batch, seq_len, _ = x.shape

        q = self.q_proj(x)  # Shape (batch_size, seq_length, d_model)
        k = self.k_proj(x)  # Shape (batch_size, seq_length, head_dim)
        v = self.v_proj(x)  # Shape (batch_size, seq_length, head_dim)

        # Reshape q to include n_heads
        q = q.reshape(x.shape[0], x.shape[1], self.n_heads, self.head_dim)  # (bqhd)

        # Ensure k and v always have a singleton n_heads dimension
        k = k.reshape(x.shape[0], x.shape[1], 1, self.head_dim)  # (bq1d)
        v = v.reshape(x.shape[0], x.shape[1], 1, self.head_dim)  # (bq1d)

        # Apply rotary embeddings to the query and key projections
        if freqs_complex is not None:
            q, k = apply_rotary_emb(q, k, freqs_complex)

        updated_kv_cache = None
        if kv_cache is not None:
            k, v, updated_kv_cache = kv_cache.update(
                xk=k,
                xv=v,
                cur_pos=start_pos,
                n_rep=self.n_heads,
            )

        attn_weights = jnp.einsum("bqhd, bkhd -> bhqk", q, k) / jnp.sqrt(self.head_dim)
        if mask is not None:
            attn_weights = jnp.where(mask[:, None, None, :], attn_weights, -1e9)

        if self.sliding_window_size is not None:
            sliding_mask = create_sliding_window_mask(
                seq_len=seq_len,
                window_size=self.sliding_window_size,
                dtype=attn_weights.dtype,
            )
            attn_weights = jnp.where(sliding_mask, attn_weights, -1e9)

        attn_weights = nnx.softmax(attn_weights, axis=-1)
        attn_weights = self.dropout(attn_weights, deterministic=deterministic)

        attn_output = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, v)
        attn_output = attn_output.reshape(x.shape[0], x.shape[1], -1)

        output = self.out_proj(attn_output)

        return output, updated_kv_cache
