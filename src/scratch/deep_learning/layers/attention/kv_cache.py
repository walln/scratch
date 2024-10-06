"""KV Cache for attention layers."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


# TODO(walln): fix layer_idx
class KVCache(NamedTuple):
    """KV Cache for attention layers.

    Standard KV cache implementation. Should work for MHA, and GQA/MQA.
    """

    k: jnp.ndarray
    v: jnp.ndarray

    @classmethod
    def create(
        cls,
        layers: int,
        bsz: int,
        max_seq_len: int,
        kv_heads: int,
        head_dim: int,
        dtype: jnp.dtype | None = None,
    ) -> "KVCache":
        """Initialize the KV cache.

        Args:
            layers: The number of layers.
            bsz: The batch size.
            max_seq_len: The maximum sequence length.
            kv_heads: The number of key and value heads.
            head_dim: The dimension of the heads.
            dtype: The dtype of the cache.
        """
        dtype = dtype or jnp.bfloat16
        return cls(
            k=jnp.zeros((layers, bsz, max_seq_len, kv_heads, head_dim), dtype=dtype),
            v=jnp.zeros((layers, bsz, max_seq_len, kv_heads, head_dim), dtype=dtype),
        )

    def update(
        self, xk: jax.Array, xv: jax.Array, layer_idx: int, cur_pos: int, n_rep: int
    ) -> tuple[jax.Array, jax.Array, "KVCache"]:
        """Update the KV cache.

        Args:
            xk: The key.
            xv: The value.
            layer_idx: The layer index.
            cur_pos: The current position.
            n_rep: The number of repetitions.
        """
        ck = jax.lax.dynamic_update_slice(
            self.k, xk[None, ...].astype(self.k.dtype), (layer_idx, 0, cur_pos, 0, 0)
        )
        cv = jax.lax.dynamic_update_slice(
            self.v, xv[None, ...].astype(self.v.dtype), (layer_idx, 0, cur_pos, 0, 0)
        )

        if cur_pos == 0:
            keys = jnp.repeat(xk, n_rep, axis=2)
            values = jnp.repeat(xv, n_rep, axis=2)
        else:
            keys = jnp.repeat(ck[layer_idx], n_rep, axis=2)
            values = jnp.repeat(cv[layer_idx], n_rep, axis=2)

        return keys, values, KVCache(k=ck, v=cv)
