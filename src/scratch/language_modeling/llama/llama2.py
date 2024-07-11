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


def precompute_theta_pos_freqs(dim: int, end: int, theta: float = 10000.0):
    """Precompute the frequency tensor for complex exponentials..

    This function calculates a frequency tensor with complex exponentials using
    the given dimension 'dim' and the end index 'end'. The 'theta' parameter
    scales the frequencies. The returned tensor contains complex values in
    complex64 data type.

    Args:
        dim: Dimension of the frequency tensor.
        end: End index for precomputing frequencies.
        theta: Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        jnp.ndarray: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)] / dim))
    t = jnp.arange(end, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)
    freqs_cis = jnp.exp(1j * freqs)  # Using Euler's formula to create complex numbers
    return freqs_cis


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
    max_batch_size: int = 32
    max_seq_len: int = 2048


class Llama2Attention(nnx.Module):
    """Llama 2 attention module.

    Llama 2 uses a grouped query attention with a kv cache. This is pretty
    much the same as the generalized grouped query attention, but it has a kv cache
    and a prefix index for prompt tokens.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        n_q_heads: int | None = None,
        max_batch_size: int = 32,
        max_seq_len: int = 2048,
        *,
        rngs: nnx.Rngs,
    ):
        """Llama 2 attention module.

        Llama 2 uses a different attention mechanism than the original Llama. The
        original Llama attention module is a multi-head attention with a single
        query and key per head. The Llama 2 attention module is a multi-head
        attention with a single query per head and a key and value per head.

        Args:
            d_model: The dimension of the model.
            n_heads: The number of attention heads.
            n_kv_heads: The number of key and value heads. Defaults to `n_heads`.
            n_q_heads: The number of query heads. Defaults to `n_heads`.
            max_batch_size: The maximum batch size. Defaults to 32.
            max_seq_len: The maximum sequence length. Defaults to 2048.
            rngs: The random number generators.
        """
        n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        n_q_heads = n_q_heads if n_q_heads is not None else n_heads
        n_rep = n_q_heads // n_kv_heads
        head_dim = d_model // n_heads

        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_rep
        self.head_dim = head_dim

        self.w_q = nnx.Linear(d_model, n_heads * head_dim, use_bias=False, rngs=rngs)
        self.w_k = nnx.Linear(d_model, n_kv_heads * head_dim, use_bias=False, rngs=rngs)
        self.w_v = nnx.Linear(d_model, n_kv_heads * head_dim, use_bias=False, rngs=rngs)
        self.w_o = nnx.Linear(n_heads * head_dim, d_model, use_bias=False, rngs=rngs)

        self.cache_k = jnp.zeros((max_batch_size, max_seq_len, n_kv_heads, head_dim))
        self.cache_v = jnp.zeros((max_batch_size, max_seq_len, n_kv_heads, head_dim))

    def __call__(
        self,
        x: jnp.ndarray,
        start_pos: int,
        freqs_complex: jnp.ndarray,
    ):
        """Compute the grouped query attention.

        Args:
            x: The input tensor.
            start_pos: The start position of the input sequence.
            freqs_complex: The frequencies for the cosine and sine functions.

        Returns:
            The output tensor.
        """
        # Get the batch size and sequence length
        batch, seq_len, _ = x.shape

        xq = self.w_q(x)
        xk = self.w_k(x)
        xv = self.w_v(x)

        xq = xq.reshape(batch, seq_len, self.n_q_heads, self.head_dim)
        xk = xk.reshape(batch, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(batch, seq_len, self.n_kv_heads, self.head_dim)

        # apply rotary embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_complex)

        # replace cache
        self.cache_k.at[:batch, start_pos : start_pos + seq_len].set(xk)
        self.cache_v.at[:batch, start_pos : start_pos + seq_len].set(xv)

        keys = self.cache_k[:batch, : start_pos + seq_len]
        values = self.cache_v[:batch, : start_pos + seq_len]

        # since each q group shares k anv v heads, we can repeat the k and v heads for
        # every q in the same group.
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose((0, 2, 1, 3))  # (batch, h_q, 1, head_dim)
        keys = keys.transpose((0, 2, 1, 3))  # (batch, h_q, kv_seq_len, head_dim)
        values = values.transpose((0, 2, 1, 3))  # (batch, h_q, kv_seq_len, head_dim)

        # Compute attention scores
        scores = xq @ keys.transpose((0, 1, 3, 2)) / jnp.sqrt(self.head_dim)
        scores = jax.nn.softmax(scores, axis=-1).astype(xq)

        output = scores @ xv
        output = output.transpose((0, 2, 1, 3)).reshape(batch, seq_len, -1)
        return self.w_o(output)


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
        self.w_1 = nnx.Linear(hidden_dim, d_model, use_bias=False, rngs=rngs)
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
        swish = jax.nn.silu(self.w_1(x))
        x_V = self.w_3(x)
        x = swish * x_V
        x = self.w_1(x)
        return x


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
        self.attn = Llama2Attention(
            config.d_model,
            config.n_heads,
            n_kv_heads=config.n_kv_heads,
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
    ):
        """Compute the transformer block forward pass.

        Args:
            x: The input tensor.
            start_pos: The start position of the input sequence.
            freqs_complex: The frequencies for the cosine and sine functions.

        Returns:
            The output tensor.
        """
        h = x + self.attn(self.attn_norm(x), start_pos, freqs_complex)
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

        self.freqs_complex = precompute_theta_pos_freqs(
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
            h = layer(h, start_pos, freqs_complex)

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
