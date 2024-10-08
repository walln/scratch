"""Multi-head attention layer.

Applies attention mechanisms to multiple query, key, and value inputs in parallel.
Each attention head computes scaled dot-product attention independently. The results
are concatenated and linearly transformed to produce the final output. This allows
the model to jointly attend to information from different representation subspaces
at different positions.

Originally proposed in [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
"""

import jax
import jax.numpy as jnp
from flax import nnx

from scratch.deep_learning.layers.attention.kv_cache import LayerKVCache
from scratch.deep_learning.layers.attention.rope import apply_rotary_emb
from scratch.deep_learning.layers.attention.sliding_window import (
    create_sliding_window_mask,
)


class MultiHeadAttention(nnx.Module):
    """Multi-head attention layer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        sliding_window_size: int | None = None,
        rngs: nnx.Rngs,
    ):
        """Initialize MultiHeadAttention.

        Args:
            d_model: The dimension of the model.
            num_heads: The number of attention heads.
            sliding_window_size: The size of the sliding window. Defaults to None.
            rngs: The random number generators.
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.sliding_window_size = sliding_window_size

        # Linear layers for query, key, value, and output
        self.w_q = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)
        self.w_k = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)
        self.w_v = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)
        self.w_o = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        freqs_complex: jnp.ndarray | None = None,
        start_pos: int = 0,
        mask: jnp.ndarray | None = None,
        kv_cache: LayerKVCache | None = None,
    ) -> tuple[jnp.ndarray, LayerKVCache | None]:
        """Compute multi-head attention forward pass.

        Args:
            x: The input tensor.
            freqs_complex: The frequencies for RoPE embeddings. Defaults to None.
            start_pos: The start position of the input sequence. Used only if
              kv_cache is provided.
            mask: The mask tensor. Defaults to None.
            kv_cache: The KV cache. Defaults to None.

        Returns:
            The output tensor and the updated KV cache.
        """
        batch, seq_len, _ = x.shape

        # Compute the query, key, and value projections
        xq = self.w_q(x)
        xk = self.w_k(x)
        xv = self.w_v(x)

        # Reshape the projections
        xq = xq.reshape(batch, seq_len, self.num_heads, self.head_dim)
        xk = xk.reshape(batch, seq_len, self.num_heads, self.head_dim)
        xv = xv.reshape(batch, seq_len, self.num_heads, self.head_dim)

        # Apply rotary embeddings to the query and key projections
        if freqs_complex is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_complex)

        # Handle KV cache if provided
        if kv_cache is not None:
            keys, values, new_kv_cache = kv_cache.update(
                xk=xk,
                xv=xv,
                cur_pos=start_pos,
                n_rep=1,
            )
        else:
            keys = xk
            values = xv
            new_kv_cache = None

        # Permute the dimensions to prepare for the attention calculation
        xq = xq.transpose((0, 2, 1, 3))  # (batch, num_heads, seq_len, head_dim)
        keys = keys.transpose((0, 2, 1, 3))  # (batch, num_heads, kv_seq_len, head_dim)
        values = values.transpose(
            (0, 2, 1, 3)
        )  # (batch, num_heads, kv_seq_len, head_dim)

        # Compute the attention scores
        scores = (xq @ keys.transpose((0, 1, 3, 2))) / jnp.sqrt(self.head_dim)
        if mask is not None:
            assert mask.ndim == 2, f"Mask must be 2D, got {mask.ndim}D"
            scores = scores + mask[None, None, :, :]

        if self.sliding_window_size is not None:
            sliding_mask = create_sliding_window_mask(
                seq_len=seq_len,
                window_size=self.sliding_window_size,
                dtype=scores.dtype,
            )
            scores = jnp.where(
                sliding_mask, scores, -1e9
            )  # Mask out positions outside the sliding window

        scores = jax.nn.softmax(scores, axis=-1).astype(xq.dtype)

        output = scores @ values
        output = output.transpose((0, 2, 1, 3)).reshape(batch, seq_len, -1)

        # Apply the output projection
        return self.w_o(output), new_kv_cache
