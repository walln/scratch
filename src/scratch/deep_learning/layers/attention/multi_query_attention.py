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
        self,
        d_model: int,
        n_heads: int,
        dropout_rate=0.1,
        use_kv_cache=True,
        max_batch_size: int = 32,
        max_seq_len: int = 2048,
        *,
        rngs: nnx.Rngs,
    ):
        """Initializes the MultiQueryAttention module.

        Args:
            d_model: The dimension of the input embeddings.
            n_heads: The number of attention heads.
            dropout_rate: The dropout rate for regularization.
            use_kv_cache: Whether to use the key-value cache. Defaults to True.
            max_batch_size: The maximum batch size. Defaults to 32.
            max_seq_len: The maximum sequence length. Defaults to 2048.
            rngs: Random number generators for initializing parameters.
        """
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self.use_kv_cache = use_kv_cache

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.q_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.k_proj = nnx.Linear(d_model, self.head_dim, rngs=rngs)
        self.v_proj = nnx.Linear(d_model, self.head_dim, rngs=rngs)
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)

        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

        if use_kv_cache:
            self.cache_k = jnp.zeros((max_batch_size, max_seq_len, self.head_dim))
            self.cache_v = jnp.zeros((max_batch_size, max_seq_len, self.head_dim))

    def __call__(self, x, mask=None, deterministic=True, start_pos=0):
        """Applies the multi-query attention mechanism.

        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model).
            mask: Attention mask of shape (batch_size, seq_length). Defaults to None.
            deterministic: If True, applies deterministic operations (e.g., no dropout).
                Defaults to True.
            start_pos: The start position of the input sequence. Used only if
                `use_kv_cache` is True.

        Returns:
            Output tensor of shape (batch_size, seq_length, d_model).
        """
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(x.shape[0], x.shape[1], self.n_heads, self.head_dim)
        k = k.reshape(x.shape[0], -1, self.head_dim)  # Shared across heads
        v = v.reshape(x.shape[0], -1, self.head_dim)  # Shared across heads

        if self.use_kv_cache:
            k, v = self.update_kv_cache(k, v, start_pos, x.shape[1], x.shape[0])

        attn_weights = jnp.einsum("...qhd,...kd->...hqk", q, k) / jnp.sqrt(
            self.head_dim
        )
        if mask is not None:
            attn_weights = jnp.where(mask[:, None, None, :], attn_weights, -1e9)

        attn_weights = nnx.softmax(attn_weights, axis=-1)
        attn_weights = self.dropout(attn_weights, deterministic=deterministic)

        attn_output = jnp.einsum("...hqk,...kd->...qhd", attn_weights, v)
        attn_output = attn_output.reshape(x.shape[0], x.shape[1], -1)

        return self.out_proj(attn_output)

    def update_kv_cache(
        self,
        keys: jnp.ndarray,
        values: jnp.ndarray,
        start_pos: int,
        seq_len: int,
        batch: int,
    ):
        """Update the KV cache.

        Args:
            keys: The keys to update the cache with.
            values: The values to update the cache with.
            start_pos: The start position of the input sequence.
            seq_len: The length of the input sequence.
            batch: The batch size.

        Returns:
            The updated keys and values.
        """
        # Update the kv cache with the new keys and values for the current sequence
        self.cache_k = self.cache_k.at[:batch, start_pos : start_pos + seq_len].set(
            keys
        )
        self.cache_v = self.cache_v.at[:batch, start_pos : start_pos + seq_len].set(
            values
        )

        # Return the updated keys and values
        out_keys = self.cache_k[:batch, : start_pos + seq_len]
        out_values = self.cache_v[:batch, : start_pos + seq_len]

        return out_keys, out_values
