"""Rotary Positional Embedding."""

import jax
import jax.numpy as jnp


def rotate_half(x: jnp.ndarray):
    """Rotate the input tensor by 90 degrees.

    Args:
        x: The input tensor.

    Returns:
        The rotated tensor.
    """
    x1, x2 = jnp.split(x, 2, axis=-1)
    x = jnp.concatenate([-x2, x1], axis=-1)
    return x


def sine_table(features: int, length: int, min_timescale=1.0, max_timescale=10000.0):
    """Create a table of sinusoidal positional embeddings.

    Args:
        features: The number of features.
        length: The length of the sequence.
        min_timescale: The minimum timescale.
        max_timescale: The maximum timescale.

    Returns:
        The sine and cosine positional embeddings table.
    """
    fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
    timescale = min_timescale * (max_timescale / min_timescale) ** fraction
    rotational_frequency = 1.0 / timescale
    # Must use high precision einsum here
    sinusoid_inp = jnp.einsum(
        "i,j->ij",
        jnp.arange(length),
        rotational_frequency,
        precision=jax.lax.Precision.HIGHEST,
    )
    sinusoid_inp = jnp.concatenate([sinusoid_inp, sinusoid_inp], axis=-1)
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def apply_rotary_pos_emb(
    q: jnp.ndarray,
    k: jnp.ndarray,
    pos_sin: jnp.ndarray,
    pos_cos: jnp.ndarray,
    index=None,
):
    """Rotary positional embeddings (RoPE).

    RoPE is a positional encoding method that uses sinusoidal embeddings and applies
    rotations to the input tensor. This method is more efficient than traditional
    sinusoidal positional embeddings and can be used in transformer models.

    For more information, see [Rotary Positional Embedding](https://arxiv.org/abs/2104.09864).

    Args:
        q: The query tensor.
        k: The key tensor.
        pos_sin: The sine positional embeddings.
        pos_cos: The cosine positional embeddings.
        index: The positional index.

    Returns:
        The rotated query and key tensors.
    """
    q_batch, q_len, q_heads, q_dim = q.shape
    k_batch, k_len, k_heads, k_dim = k.shape

    if index is not None:
        q_cos = jax.lax.broadcast_in_dim(
            pos_cos[index, :], (q_batch, q_len, q_heads, q_dim), (3,)
        )
        q_sin = jax.lax.broadcast_in_dim(
            pos_sin[index, :], (q_batch, q_len, q_heads, q_dim), (3,)
        )
    else:
        q_cos = jax.lax.broadcast_in_dim(
            pos_cos[:q_len, :], (q_batch, q_len, q_heads, q_dim), (1, 3)
        )
        q_sin = jax.lax.broadcast_in_dim(
            pos_sin[:q_len, :], (q_batch, q_len, q_heads, q_dim), (1, 3)
        )

    k_cos = jax.lax.broadcast_in_dim(
        pos_cos[:k_len, :], (k_batch, k_len, k_heads, k_dim), (1, 3)
    )
    k_sin = jax.lax.broadcast_in_dim(
        pos_sin[:k_len, :], (k_batch, k_len, k_heads, k_dim), (1, 3)
    )

    q_rot = (q_cos * q) + (q_sin * rotate_half(q))
    k_rot = (k_cos * k) + (k_sin * rotate_half(k))
    return q_rot, k_rot
