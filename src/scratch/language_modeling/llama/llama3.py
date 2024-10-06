"""LLama 3 Implementation.

This file contains the implementation of the 3rd generation LLama (Large Language Model)
model from Meta AI. LLama is a transformer-based language model that is trained on a
large corpus of text data using unsupervised learning techniques. It is designed to be
highly efficient and scalable, making it suitable for a wide range of natural language
processing tasks.

The implementation is based on the paper "LLaMA: Open and Efficient Foundation Language
Models" by Meta AI.

Reference:
    Meta AI: LLaMA: Open and Efficient Foundation Language Models
    arXiv preprint arXiv:2302.13971. https://arxiv.org/abs/2302.13971

Llama 3 primarily differs from Llama 2 in that it uses a different attention mechanism
and a different set of default hyperparameters including a much more efficient
tokenizer. Other than that, the differences are during training, largely in the
much larger and more diverse training corpus: 15 TRILLION tokens!
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from scratch.deep_learning.layers.attention.grouped_query_attention import (
    GroupedQueryAttention,
)
from scratch.deep_learning.layers.attention.kv_cache import KVCache


@dataclass
class Llama3Config:
    """Configuration for the LLama 3 model.

    Attributes:
        d_model: Size of the model.
        n_layers: Number of layers.
        n_heads: Number of attention heads.
        vocab_size: Size of the vocabulary.
        n_kv_heads: Number of key-value heads.
        multiple_of: Make SwiGLU hidden layer size multiple of large power of 2.
        ffn_dim_multiplier: Multiplier for the hidden dimension of the feed-forward
          layer.
        norm_eps: Epsilon value for layer normalization.
        rope_theta: RoPE theta value.
        max_batch_size: Maximum batch size.
        max_seq_len: Maximum sequence length.
    """

    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = 128256
    n_kv_heads: int | None = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048


class LLama3MLP(nnx.Module):
    """MLP with dimensional multiplication."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: float | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the LLama 3 MLP.

        Args:
            d_model: The model dimension.
            d_ff: The feed-forward dimension.
            multiple_of: The multiple of for the hidden dimension.
            ffn_dim_multiplier: The ffn_dim_multiplier for the hidden dimension.
            rngs: The random number generators.
        """
        hidden_dim = int(2 * d_ff / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w_1 = nnx.Linear(d_model, hidden_dim, use_bias=False, rngs=rngs)
        self.w_2 = nnx.Linear(hidden_dim, d_model, use_bias=False, rngs=rngs)
        self.w_3 = nnx.Linear(d_model, hidden_dim, use_bias=False, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
    ):
        """Compute the MLP forward pass.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        return self.w_2(jax.nn.silu(self.w_1(x)) * self.w_3(x))


class LLama3TransformerBlock(nnx.Module):
    """Transformer block with Llama 3 attention and MLP."""

    def __init__(
        self,
        config: Llama3Config,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the transformer block.

        Args:
            config: The configuration for the model.
            rngs: The random number generators.
        """
        self.attn = GroupedQueryAttention(
            config.d_model,
            config.n_heads,
            n_kv_heads=config.n_kv_heads,
            rngs=rngs,
        )
        self.mlp = LLama3MLP(
            config.d_model,
            config.d_model * 4,
            ffn_dim_multiplier=config.ffn_dim_multiplier,
            multiple_of=config.multiple_of,
            rngs=rngs,
        )
        self.attn_norm = nnx.RMSNorm(config.d_model, epsilon=config.norm_eps, rngs=rngs)
        self.mlp_norm = nnx.RMSNorm(config.d_model, epsilon=config.norm_eps, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        start_pos: int,
        freqs_complex: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        kv_cache: KVCache | None = None,
    ) -> tuple[jnp.ndarray, KVCache | None]:
        """Compute the transformer block forward pass.

        Args:
            x: The input tensor.
            start_pos: The start position of the input sequence.
            freqs_complex: The frequencies for the cosine and sine functions.
            mask: The mask for the attention.
            kv_cache: The KV cache. Defaults to None.

        Returns:
            The output tensor and the updated KV cache.
        """
        h, new_kv_cache = self.attn(
            self.attn_norm(x),
            start_pos=start_pos,
            freqs_complex=freqs_complex,
            mask=mask,
            kv_cache=kv_cache,
        )
        h = x + h
        out = h + self.mlp(self.mlp_norm(h))
        return out, new_kv_cache


class LLama3(nnx.Module):
    """LLama 3 model."""

    def __init__(
        self,
        config: Llama3Config,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the LLama 3 model.

        Args:
            config: The configuration for the model.
            rngs: The random number generators.
        """
        self.tok_embeddings = nnx.Embed(
            config.vocab_size,
            config.d_model,
            rngs=rngs,
        )
        self.layers = [
            LLama3TransformerBlock(config, rngs=rngs) for _ in range(config.n_layers)
        ]

        self.norm = nnx.RMSNorm(config.d_model, epsilon=config.norm_eps, rngs=rngs)
        self.output = nnx.Linear(
            config.d_model, config.vocab_size, use_bias=False, rngs=rngs
        )

        self.freqs_complex = GroupedQueryAttention.precompute_theta_pos_freqs(
            config.d_model // config.n_heads, config.max_seq_len * 2, config.rope_theta
        )

    def __call__(
        self,
        x: jnp.ndarray,
        start_pos: int,
        kv_cache: list[KVCache] | None = None,
    ) -> tuple[jnp.ndarray, list[KVCache | None]]:
        """Compute the LLama 3 forward pass.

        Args:
            x: The input tensor.
            start_pos: The start position of the input sequence.
            kv_cache: The KV cache for each layer. Defaults to None.

        Returns:
            The output tensor and the updated KV cache for each layer.
        """
        batch, seq_len = x.shape
        h = self.tok_embeddings(x)
        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]

        mask = None
        if seq_len > 1:
            mask = jnp.full((seq_len, seq_len), -jnp.inf)
            mask = jnp.triu(mask, k=1)
            mask = jnp.hstack([jnp.zeros((seq_len, start_pos)), mask]).astype(h.dtype)

        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            layer_kv_cache = kv_cache[i] if kv_cache is not None else None
            h, new_layer_kv_cache = layer(
                h,
                start_pos=start_pos,
                freqs_complex=freqs_complex,
                mask=mask,
                kv_cache=layer_kv_cache,
            )
            new_kv_cache.append(new_layer_kv_cache)

        h = self.norm(h)
        output = self.output(h).astype(jnp.float32)
        return output, new_kv_cache


if __name__ == "__main__":
    config = Llama3Config(
        d_model=256,
        n_layers=4,
        n_heads=4,
        vocab_size=256,
        max_seq_len=128,
        max_batch_size=2,
    )
    model = LLama3(config, rngs=nnx.Rngs(0))
    kv_caches = [
        KVCache.create(
            config.n_layers,
            config.max_batch_size,
            config.max_seq_len,
            config.n_heads,
            config.d_model,
        )
        for _ in range(config.n_layers)
    ]
    x = jnp.ones((1, config.max_seq_len), dtype=jnp.int32)
    y, _ = model(x, 0, kv_caches)
    print(y.shape)
