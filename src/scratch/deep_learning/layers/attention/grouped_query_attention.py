"""Grouped Query Attention (GQA).

This module implements Grouped Query Attention as described in "GQA: Training
Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" by
Joshua Ainslie et al. The main idea is to use a reduced number of key-value
heads compared to query heads, striking a balance between computational
efficiency and model quality.

GQA can be seen as a generalization of Multi-Query Attention (MQA) and Multi-Head
Attention (MHA), allowing for flexible grouping of query heads that share the same
key and value heads. This approach maintains most of the modeling power of MHA while
significantly reducing computation and memory requirements.

Reference:
    Ainslie, Joshua, et al. "GQA: Training Generalized Multi-Query Transformer Models
    from Multi-Head Checkpoints."
    arXiv preprint arXiv:2305.13245 (2023). https://arxiv.org/abs/2305.13245
"""

import jax
import jax.numpy as jnp
from flax import nnx

from scratch.deep_learning.layers.attention.kv_cache import LayerKVCache
from scratch.deep_learning.layers.attention.rope import apply_rotary_emb, repeat_kv
from scratch.deep_learning.layers.attention.sliding_window import (
    create_sliding_window_mask,
)


class GroupedQueryAttention(nnx.Module):
    """Grouped query attention module with optional KV cache."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        n_q_heads: int | None = None,
        *,
        sliding_window_size: int | None = None,
        rngs: nnx.Rngs,
    ):
        """Initialize the Llama 2 attention module.

        Args:
            d_model: The dimension of the model.
            n_heads: The number of attention heads.
            n_kv_heads: The number of key and value heads. Defaults to `n_heads`.
            n_q_heads: The number of query heads. Defaults to `n_heads`.
            sliding_window_size: The size of the sliding window. Defaults to None.
            rngs: The random number generators.
        """
        n_q_heads = n_heads if n_q_heads is None else n_q_heads
        n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        n_rep = n_q_heads // n_kv_heads
        head_dim = d_model // n_heads

        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_rep
        self.head_dim = head_dim

        self.sliding_window_size = sliding_window_size

        self.w_q = nnx.Linear(d_model, n_heads * head_dim, use_bias=False, rngs=rngs)
        self.w_k = nnx.Linear(d_model, n_kv_heads * head_dim, use_bias=False, rngs=rngs)
        self.w_v = nnx.Linear(d_model, n_kv_heads * head_dim, use_bias=False, rngs=rngs)
        self.w_o = nnx.Linear(n_heads * head_dim, d_model, use_bias=False, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        freqs_complex: jnp.ndarray | None = None,
        start_pos: int = 0,
        mask: jnp.ndarray | None = None,
        kv_cache: LayerKVCache | None = None,
    ) -> tuple[jnp.ndarray, LayerKVCache | None]:
        """Compute the grouped query attention.

        Args:
            x: The input tensor.
            freqs_complex: The frequencies for RoPE embeddings. Defaults to None.
            start_pos: The start position of the input sequence. Used only if
              kv_cache is provided.
            mask: The mask for the attention. Defaults to None.
            kv_cache: The KV cache. Defaults to None.

        Returns:
            The output tensor and the updated KV cache.
        """
        batch, seq_len, _ = x.shape

        # Compute the query, key, and value projections.
        xq = self.w_q(x)
        xk = self.w_k(x)
        xv = self.w_v(x)

        # Reshape the projections based on the group size
        xq = xq.reshape(batch, seq_len, self.n_q_heads, self.head_dim)
        xk = xk.reshape(batch, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Apply rotary embeddings to the query and key projections
        if freqs_complex is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_complex)

        # Handle KV cache if provided
        if kv_cache is not None:
            keys, values, new_kv_cache = kv_cache.update(
                xk=xk,
                xv=xv,
                cur_pos=start_pos,
                n_rep=self.n_rep,
            )
        else:
            keys = repeat_kv(xk, self.n_rep)
            values = repeat_kv(xv, self.n_rep)
            new_kv_cache = None

        # Permute the dimensions to prepare for the attention calculation
        xq = xq.transpose((0, 2, 1, 3))  # (batch, h_q, seq_len, head_dim)
        keys = keys.transpose((0, 2, 1, 3))  # (batch, h_q, kv_seq_len, head_dim)
        values = values.transpose((0, 2, 1, 3))  # (batch, h_q, kv_seq_len, head_dim)

        # Compute the attention scores
        scores = xq @ keys.transpose((0, 1, 3, 2)) / jnp.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask

        if self.sliding_window_size is not None:
            sliding_mask = create_sliding_window_mask(
                seq_len=seq_len,
                window_size=self.sliding_window_size,
                dtype=scores.dtype,
            )
            scores = jnp.where(sliding_mask, scores, -1e9)

        scores = jax.nn.softmax(scores, axis=-1).astype(xq.dtype)

        output = scores @ values
        output = output.transpose((0, 2, 1, 3)).reshape(batch, seq_len, -1)

        # Apply the output projection
        return self.w_o(output), new_kv_cache
