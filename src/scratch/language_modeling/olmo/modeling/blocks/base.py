"""Base OLMo transformer block."""

import jax.numpy as jnp
from flax import nnx

from scratch.language_modeling.olmo.modeling.activations import SwiGLU
from scratch.language_modeling.olmo.modeling.config import OLMoConfig
from scratch.language_modeling.olmo.modeling.rotary_embedding import (
    apply_rotary_pos_emb,
    sine_table,
)


def spda(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    mask: jnp.ndarray | None = None,
    dropout: float = 0.0,
    is_causal=False,
    deterministic=False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate the scaled dot-product attention with causal masking and dropout.

    Args:
        q: Query tensor of shape (..., seq_len_q, depth).
        k: Key tensor of shape (..., seq_len_k, depth).
        v: Value tensor of shape (..., seq_len_v, depth_v).
        mask: Float tensor of shape (..., seq_len_q, seq_len_k). Optional.
        is_causal: Whether to apply causal masking. Default is False.
        dropout: Dropout rate to apply to the attention weights. Default is 0.0.
        deterministic: Whether to apply dropout deterministically. Default is False.

    Returns:
        Output tensor of shape (..., seq_len_q, depth_v).
        Attention weights tensor of shape (..., seq_len_q, seq_len_k).
    """
    matmul_qk = jnp.einsum("...ij,...kj->...ik", q, k)

    # Scale matmul_qk
    depth = q.shape[-1]
    logits = matmul_qk / jnp.sqrt(depth)

    # Apply causal mask
    if is_causal:
        seq_len_q, seq_len_k = q.shape[-2], k.shape[-2]
        causal_mask = jnp.tril(jnp.ones((seq_len_q, seq_len_k), dtype=jnp.float32))
        logits = jnp.where(causal_mask == 0, -1e9, logits)

    # Add the mask to the scaled tensor.
    if mask is not None:
        logits += mask * -1e9

    # Softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = nnx.softmax(logits, axis=-1)

    # Apply dropout
    if dropout > 0.0:
        attention_weights = nnx.Dropout(rate=dropout, deterministic=deterministic)(
            attention_weights
        )

    output = jnp.einsum("...ij,...jk->...ik", attention_weights, v)

    return output, attention_weights


class Block(nnx.Module):
    """Base OLMo transformer block."""

    q_norm: nnx.LayerNorm | None
    k_norm: nnx.LayerNorm | None

    def __init__(self, layer_id: int, config: OLMoConfig, *, rngs: nnx.Rngs):
        """Initialize the OLMo block.

        Args:
            layer_id: the ID of the layer
            config: the model configuration
            rngs: the random number generators
        """
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = config.max_sequence_length * 3
        self.activation_multiplier = 0.5  # SwiGLU activation multiplier
        self.q_norm = None
        self.k_norm = None

        # Ensure the model dimensions are valid
        assert config.d_model % config.n_heads == 0

        self.dropout = nnx.Dropout(config.residual_dropout, rngs=rngs)

        # Make sure QKV clip coefficient is positive, otherwise it's not well-defined.
        if config.clip_qkv is not None:
            assert config.clip_qkv > 0

        self.activation = SwiGLU()
        assert (self.activation_multiplier * self.hidden_size) % 1 == 0

        # Compute attention projections
        self.attention_out = nnx.Linear(
            config.d_model, config.d_model, use_bias=config.include_bias, rngs=rngs
        )

        # Feed-forward network
        self.ff_out = nnx.Linear(
            int(self.activation_multiplier * self.hidden_size),
            config.d_model,
            use_bias=config.include_bias,
            rngs=rngs,
        )

    def _scaled_dot_product_attention(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
        dropout: float = 0.0,
        is_causal=False,
    ) -> jnp.ndarray:
        """Computes scaled dot-product attention.

        Args:
            q: the query tensor
            k: the key tensor
            v: the value tensor
            attention_mask: the attention mask tensor
            dropout: the dropout rate
            is_causal: whether the attention is causal
        """
        # jax's sdpa doesn't support GQA, so we're doing this
        assert k.shape[1] == v.shape[1]
        num_kv_heads = k.shape[1]
        num_q_heads = q.shape[1]
        if num_q_heads != num_kv_heads:
            assert num_q_heads % num_kv_heads == 0
            k = k.repeat(num_q_heads // num_kv_heads, axis=1)
            v = v.repeat(num_q_heads // num_kv_heads, axis=1)

        output, attention_weights = spda(
            q,
            k,
            v,
            mask=attention_mask,
            dropout=dropout,
            is_causal=is_causal,
        )

        return output

    def attention(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        attention_bias: jnp.ndarray | None = None,
        layer_past: tuple[jnp.ndarray, jnp.ndarray] | None = None,
        training=False,
    ) -> jnp.ndarray:
        """Compute the attention mechanism.

        Args:
            q: The query tensor.
            k: The key tensor.
            v: The value tensor.
            attention_bias: The attention bias tensor.
            layer_past: The past layer tensor.
            training: Whether the model is in training mode.

        Returns:
            The output tensor.
        """
        B, T, C = q.shape
        dtype = k.dtype

        # apply layer norm if needed
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).astype(dtype)
            k = self.k_norm(k).astype(dtype)

        print(f"Q shape: {q.shape}")
        print(f"K shape: {k.shape}")
        print(f"V shape: {v.shape}")

        # Move head forward to be next to the batch dim.
        # shape: (B, nh, T, hs)
        q = q.reshape(B, T, self.config.n_heads, C // self.config.n_heads).transpose(
            0, 2, 1, 3
        )
        # shape: (B, n_kv_h, T, hs)
        k = k.reshape(
            B, T, self.config.effective_n_kv_heads, C // self.config.n_heads
        ).transpose(0, 2, 1, 3)
        # shape: (B, n_kv_h, T, hs)
        v = v.reshape(
            B, T, self.config.effective_n_kv_heads, C // self.config.n_heads
        ).transpose(0, 2, 1, 3)

        print(f"Attention: q.shape: {q.shape}")
        print(f"Attention: k.shape: {k.shape}")
        print(f"Attention: v.shape: {v.shape}")

        if layer_past is not None:
            past_key, past_value = layer_past
            k = jnp.concatenate((past_key, k), axis=-2)
            v = jnp.concatenate((past_value, v), axis=-2)

        if self.config.use_rope:
            # Apply rotary embeddings
            sin, cos = sine_table(q.shape[-1], max(q.shape[1], k.shape[1]))
            q, k = apply_rotary_pos_emb(q, k, sin, cos)

        attention = self._scaled_dot_product_attention(
            q,
            k,
            v,
            attention_mask=attention_bias,
            dropout=0.0 if not training else self.config.attention_dropout,
            is_causal=attention_bias is None,
        )

        # put the outputs of the heads together again
        attention = attention.transpose(0, 2, 1, 3).reshape(B, T, C)

        # Output projection
        return self.attention_out(attention)
