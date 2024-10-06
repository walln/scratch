"""LLama 2 Implementation.

This file contains the implementation of the 2nd generation LLama (Large Language Model)
model from Meta AI. LLama is a transformer-based language model that is trained on a
large corpus of text data using unsupervised learning techniques. It is designed to be
highly efficient and scalable, making it suitable for a wide range of natural language
processing tasks.

The implementation is based on the paper "LLaMA: Open and Efficient Foundation Language
Models" by Meta AI.

Reference:
    Meta AI: LLaMA: Open and Efficient Foundation Language Models
    arXiv preprint arXiv:2302.13971. https://arxiv.org/abs/2302.13971

Llama 2 primarily differs from Llama 1 in that it uses Grouped Query Attention (GQA)
and a larger context length. The other hyperparameters remain largely the same. The
main difference is the attention to data, the higher quality data, larger corpus, and
implementation of RLHF during training dramatically improves performance.


Overall, Llama 2 represents a significant advancement over Llama 1, offering improved
efficiency, scalability, and performance across a wide range of natural language
processing tasks.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from scratch.deep_learning.layers.attention.grouped_query_attention import (
    GroupedQueryAttention,
)


@dataclass
class Llama2Config:
    """Configuration for the LLama 2 model.

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
    n_kv_heads: int | None = None
    vocab_size: int = 32000
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5

    # Needed for KV cache
    use_kv_cache: bool = True
    max_batch_size: int = 32
    max_seq_len: int = 2048


class Llama2MLP(nnx.Module):
    """Llama 2 MLP module.

    Llama 2 uses a mlp with a hidden dimension multiplier.

    Args:
        d_model: The dimension of the model.
        d_ff: The feed-forward dimension.
        multiple_of: The multiple of for the hidden dimension.
        ffn_dim_multiplier: The ffn_dim_multiplier for the hidden dimension.
        rngs: The random number generators.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: float | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        """Llama 2 MLP module.

        Llama 2 uses a mlp with a hidden dimension multiplier.

        Args:
            d_model: The dimension of the model.
            d_ff: The feed-forward dimension.
            multiple_of: The multiple of for the hidden dimension.
            ffn_dim_multiplier: The ffn_dim_multiplier for the hidden dimension.
            rngs: The random number generators.
        """
        hidden_dim = 4 * d_model
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
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


class Llama2TransformerBlock(nnx.Module):
    """Transformer block with Llama 2 attention and MLP."""

    def __init__(
        self,
        config: Llama2Config,
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
            use_kv_cache=config.use_kv_cache,
            max_batch_size=config.max_batch_size,
            max_seq_len=config.max_seq_len,
            rngs=rngs,
        )
        self.mlp = Llama2MLP(
            config.d_model,
            config.d_model * 4,
            multiple_of=config.multiple_of,
            ffn_dim_multiplier=config.ffn_dim_multiplier,
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
    ):
        """Compute the transformer block forward pass.

        Args:
            x: The input tensor.
            start_pos: The start position of the input sequence.
            freqs_complex: The frequencies for the cosine and sine functions.
            mask: The mask for the attention. Defaults to None.

        Returns:
            The output tensor.
        """
        h = x + self.attn(
            self.attn_norm(x),
            start_pos=start_pos,
            freqs_complex=freqs_complex,
            mask=mask,
        )
        out = h + self.mlp(self.mlp_norm(h))
        return out


class Llama2(nnx.Module):
    """Llama 2 model."""

    def __init__(
        self,
        config: Llama2Config,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the Llama 2 model.

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
            Llama2TransformerBlock(config, rngs=rngs) for _ in range(config.n_layers)
        ]

        self.norm = nnx.RMSNorm(config.d_model, epsilon=config.norm_eps, rngs=rngs)
        self.output = nnx.Linear(
            config.d_model, config.vocab_size, use_bias=False, rngs=rngs
        )

        self.freqs_complex = GroupedQueryAttention.precompute_theta_pos_freqs(
            config.d_model // config.n_heads, config.max_seq_len * 2
        )

    def __call__(
        self,
        x: jnp.ndarray,
        start_pos: int,
    ):
        """Compute the Llama 2 forward pass.

        Args:
            x: The input tensor.
            start_pos: The start position of the input sequence.

        Returns:
            The output tensor.
        """
        batch, seq_len = x.shape
        h = self.tok_embeddings(x)
        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]

        for layer in self.layers:
            h = layer(
                h,
                start_pos=start_pos,
                freqs_complex=freqs_complex,
            )

        h = self.norm(h)
        output = self.output(h).astype(jnp.float32)
        return output


if __name__ == "__main__":
    config = Llama2Config(
        d_model=256,
        n_layers=4,
        n_heads=4,
        vocab_size=256,
        max_seq_len=128,
        max_batch_size=2,
    )
    model = Llama2(config, rngs=nnx.Rngs(0))
    x = jnp.ones((1, config.max_seq_len), dtype=jnp.int32)
    y = model(x, 0)
    print(y.shape)
