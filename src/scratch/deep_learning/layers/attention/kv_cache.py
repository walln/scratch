"""KV Cache for attention layers."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class LayerKVCache(NamedTuple):
    """KV Cache for attention layers.

    Standard KV cache implementation. Should work for MHA, and GQA/MQA.
    """

    k: jnp.ndarray
    v: jnp.ndarray

    @classmethod
    def create(
        cls,
        bsz: int,
        max_seq_len: int,
        kv_heads: int,
        head_dim: int,
        dtype: jnp.dtype | None = None,
    ) -> "LayerKVCache":
        """Initialize the KV cache.

        Args:
            bsz: The batch size.
            max_seq_len: The maximum sequence length.
            kv_heads: The number of key and value heads.
            head_dim: The dimension of the heads.
            dtype: The dtype of the cache.
        """
        dtype = dtype or jnp.bfloat16
        return cls(
            k=jnp.zeros((bsz, max_seq_len, kv_heads, head_dim), dtype=dtype),
            v=jnp.zeros((bsz, max_seq_len, kv_heads, head_dim), dtype=dtype),
        )

    def update(
        self, xk: jax.Array, xv: jax.Array, cur_pos: int, n_rep: int
    ) -> tuple[jax.Array, jax.Array, "LayerKVCache"]:
        """Update the KV cache.

        Args:
            xk: The key.
            xv: The value.
            cur_pos: The current position.
            n_rep: The number of repetitions.
        """
        ck = jax.lax.dynamic_update_slice(
            self.k, xk.astype(self.k.dtype), (0, cur_pos, 0, 0)
        )
        cv = jax.lax.dynamic_update_slice(
            self.v, xv.astype(self.v.dtype), (0, cur_pos, 0, 0)
        )

        if cur_pos == 0:
            keys = jnp.repeat(xk, n_rep, axis=2)
            values = jnp.repeat(xv, n_rep, axis=2)
        else:
            keys = jnp.repeat(ck, n_rep, axis=2)
            values = jnp.repeat(cv, n_rep, axis=2)

        return keys, values, LayerKVCache(k=ck, v=cv)


class KVCache:
    """KV cache that supports multiple layers or experts."""

    def __init__(self, layer_caches: list[LayerKVCache]):
        """Initialize the multi-layer KV cache.

        Args:
            layer_caches: A list of LayerKVCache objects, one for each layer or expert.
        """
        self.layer_caches = layer_caches

    @classmethod
    def create(
        cls,
        num_layers: int,
        bsz: int,
        max_seq_len: int,
        kv_heads: int,
        head_dim: int,
        dtype: jnp.dtype | None = None,
    ) -> "KVCache":
        """Create a new KVCache.

        Args:
            num_layers: The number of layers or experts.
            bsz: The batch size.
            max_seq_len: The maximum sequence length.
            kv_heads: The number of key and value heads.
            head_dim: The dimension of the heads.
            dtype: The dtype of the cache.

        Returns:
            A new MultiLayerKVCache instance.
        """
        layer_caches = [
            LayerKVCache.create(
                bsz=bsz,
                max_seq_len=max_seq_len,
                kv_heads=kv_heads,
                head_dim=head_dim,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ]
        return cls(layer_caches)

    def update(
        self,
        layer_idx: int,
        xk: jax.Array,
        xv: jax.Array,
        cur_pos: int,
        n_rep: int,
    ) -> tuple[jax.Array, jax.Array, "KVCache"]:
        """Update the KV cache for a specific layer.

        Args:
            layer_idx: The index of the layer to update.
            xk: The key.
            xv: The value.
            cur_pos: The current position.
            n_rep: The number of repetitions.

        Returns:
            A tuple containing the updated keys, values, and the new KVCache.
        """
        keys, values, updated_layer_cache = self.layer_caches[layer_idx].update(
            xk=xk, xv=xv, cur_pos=cur_pos, n_rep=n_rep
        )

        new_layer_caches = self.layer_caches.copy()
        new_layer_caches[layer_idx] = updated_layer_cache

        return keys, values, KVCache(new_layer_caches)

    def __getitem__(self, idx: int) -> LayerKVCache:
        """Get the LayerKVCache for a specific layer.

        Args:
            idx: The index of the layer.

        Returns:
            The LayerKVCache for the specified layer.
        """
        return self.layer_caches[idx]

    def __setitem__(self, idx: int, value: LayerKVCache) -> None:
        """Update the LayerKVCache for a specific layer.

        Args:
            idx: The index of the layer to update.
            value: The new LayerKVCache.
        """
        self.layer_caches[idx] = value

    def __len__(self) -> int:
        """Get the number of layers in the cache.

        Returns:
            The number of layers in the cache.
        """
        return len(self.layer_caches)
