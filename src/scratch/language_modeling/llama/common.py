"""Common utilities for LLaMA."""

import jax
import jax.numpy as jnp


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
