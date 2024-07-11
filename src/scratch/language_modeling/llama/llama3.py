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
import numpy as np
from flax import nnx


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

    max_batch_size: int = 32
    max_seq_len: int = 2048


def apply_rotary_emb(
    xq: jnp.ndarray,
    xk: jnp.ndarray,
    freqs_cis: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply rotary embedding to the query and key tensors.

    Args:
        xq: The query tensor.
        xk: The key tensor.
        freqs_cis: The frequencies for the cosine and sine functions.

    Returns:
        The rotated query and key tensors.
    """
    xq_ = jax.lax.complex(
        xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)[..., 0],
        xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)[..., 1],
    )
    xk_ = jax.lax.complex(
        xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)[..., 0],
        xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)[..., 1],
    )

    freqs_cis = jnp.expand_dims(freqs_cis, axis=(0, 2))

    xq_out = jnp.stack(
        [jnp.real(xq_ * freqs_cis), jnp.imag(xq_ * freqs_cis)], axis=-1
    ).reshape(*xq.shape)
    xk_out = jnp.stack(
        [jnp.real(xk_ * freqs_cis), jnp.imag(xk_ * freqs_cis)], axis=-1
    ).reshape(*xk.shape)

    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)


def repeat_kv(x: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    """Repeats the key/value tensor along the head dimension.

    Args:
        x (batch, seq_len, n_kv_heads, head_dim): The input tensor.
        n_rep: The number of repetitions for each key/value head.

    Returns:
        The repeated tensor (batch, seq_len, n_kv_heads * n_rep, head_dim).
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return jnp.tile(x[:, :, :, None, :], (1, 1, 1, n_rep, 1)).reshape(
        bs, slen, n_kv_heads * n_rep, head_dim
    )


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute frequencies for RoPE.

    Args:
        dim: Dimension of the model.
        end: End of the range.
        theta: Theta for RoPE.

    Returns:
        The frequencies.
    """
    freqs = 1.0 / (
        theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim)
    )
    t = jnp.arange(end, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)
    freqs_cis = jnp.exp(1j * freqs)  # Using Euler's formula to create complex numbers
    return freqs_cis


class LLama3GroupedQueryAttention(nnx.Module):
    """Grouped query attention with a kv cache.

    Llama 3 uses a grouped query attention with a kv cache. This is pretty
    much the same as the generalized grouped query attention, but it has a kv cache
    and a prefix index for prompt tokens.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        max_batch_size: int = 32,
        max_seq_len: int = 2048,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the grouped query attention module.

        Args:
            d_model: The model dimension.
            n_heads: The number of heads.
            n_kv_heads: The number of kv heads. If None, defaults to n_heads.
            max_batch_size: The maximum batch size. Defaults to 32.
            max_seq_len: The maximum sequence length. Defaults to 2048.
            rngs: The random number generators.
        """
        n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        head_dim = d_model // n_heads

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.w_q = nnx.Linear(d_model, n_heads * head_dim, use_bias=False, rngs=rngs)
        self.w_k = nnx.Linear(d_model, n_kv_heads * head_dim, use_bias=False, rngs=rngs)
        self.w_v = nnx.Linear(d_model, n_kv_heads * head_dim, use_bias=False, rngs=rngs)
        self.w_o = nnx.Linear(n_heads * head_dim, d_model, use_bias=False, rngs=rngs)

        self.cache_k = jnp.zeros(
            (
                max_batch_size,
                max_seq_len,
                n_kv_heads,
                head_dim,
            )
        )
        self.cache_v = jnp.zeros(
            (
                max_batch_size,
                max_seq_len,
                n_kv_heads,
                head_dim,
            )
        )

    def __call__(
        self,
        x: jnp.ndarray,
        start_pos: int,
        freqs_cis: jnp.ndarray,
        mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute the grouped query attention.

        Args:
            x: The input tensor.
            start_pos: The start position of the input sequence.
            freqs_cis: The frequencies for the cosine and sine functions.
            mask: The mask for the attention.

        Returns:
            The output tensor.
        """
        # Get the batch size and sequence length
        batch, seq_len, _ = x.shape

        # Get the query, key, and value vectors
        xq = self.w_q(x)
        xk = self.w_k(x)
        xv = self.w_v(x)

        # Compute the attention weights
        xq = xq.reshape(batch, seq_len, self.n_heads, self.head_dim)
        xk = xk.reshape(batch, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(batch, seq_len, self.n_kv_heads, self.head_dim)

        # apply rotary embedding
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        self.cache_k.at[:batch, start_pos : start_pos + seq_len].set(xk)
        self.cache_v.at[:batch, start_pos : start_pos + seq_len].set(xv)

        keys = self.cache_k[:batch, : start_pos + seq_len]
        values = self.cache_v[:batch, : start_pos + seq_len]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_heads // self.n_kv_heads)
        values = repeat_kv(values, self.n_heads // self.n_kv_heads)

        xq = xq.transpose((0, 2, 1, 3))  # (batch, n_heads, seq_len, head_dim)
        keys = keys.transpose(
            (0, 2, 1, 3)
        )  # (batch, n_heads, cache_len + seq_len, head_dim)
        values = values.transpose(
            (0, 2, 1, 3)
        )  # (batch, n_heads, cache_len + seq_len, head_dim)

        # compute attention scores
        scores = xq @ keys.transpose((0, 1, 3, 2)) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask

        scores = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)
        output = scores @ values  # (batch, n_heads, seq_len, head_dim)
        output = output.transpose((0, 2, 1, 3))  # (batch, seq_len, n_heads, head_dim)
        output = output.reshape(batch, seq_len, -1)
        return self.w_o(output)


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
        self.attn = LLama3GroupedQueryAttention(
            config.d_model,
            config.n_heads,
            n_kv_heads=config.n_kv_heads,
            max_batch_size=config.max_batch_size,
            max_seq_len=config.max_seq_len,
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
        freqs_cis: jnp.ndarray,
        mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute the transformer block forward pass.

        Args:
            x: The input tensor.
            start_pos: The start position of the input sequence.
            freqs_cis: The frequencies for the cosine and sine functions.
            mask: The mask for the attention.

        Returns:
            The output tensor.
        """
        h = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        out = h + self.mlp(self.mlp_norm(h))
        return out


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

        self.freqs_cis = precompute_freqs_cis(
            config.d_model // config.n_heads, config.max_seq_len * 2, config.rope_theta
        )

    def __call__(
        self,
        x: jnp.ndarray,
        start_pos: int,
        mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute the LLama 3 forward pass.

        Args:
            x: The input tensor.
            start_pos: The start position of the input sequence.
            mask: The mask for the attention.

        Returns:
            The output tensor.
        """
        batch, seq_len = x.shape
        h = self.tok_embeddings(x)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]

        mask = None
        if seq_len > 1:
            mask = jnp.full((seq_len, seq_len), -jnp.inf)

            mask = jnp.triu(mask, k=1)

            mask = jnp.hstack([jnp.zeros((seq_len, start_pos)), mask]).astype(h.dtype)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).astype(jnp.float32)
        return output


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
    x = jnp.ones((1, config.max_seq_len), dtype=jnp.int32)
    y = model(x, 0)
    print(y.shape)
