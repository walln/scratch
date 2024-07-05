"""Sequential Block."""

import jax.numpy as jnp
from flax import nnx

from scratch.language_modeling.olmo.modeling.blocks.base import Block
from scratch.language_modeling.olmo.modeling.config import OLMoConfig


class SequentialBlock(Block):
    """Standard OLMo transformer block.

    Computes: ``MLP(LN(x + Attention(LN(x))))`` and adds a residual connection.
    """

    def __init__(self, layer_id: int, config: OLMoConfig, *, rngs: nnx.Rngs):
        """Initialize the sequential OLMo block.

        Args:
            layer_id: the ID of the layer
            config: the model configuration
            rngs: the random number generators
        """
        super().__init__(layer_id, config, rngs=rngs)

        # Create layer norms
        self.attn_norm = nnx.LayerNorm(config.d_model, rngs=rngs)
        self.ff_norm = nnx.LayerNorm(config.d_model, rngs=rngs)

        # Attention projections
        head_dim = config.d_model // config.n_heads
        self.fused_dims = (
            config.d_model,
            config.effective_n_kv_heads * head_dim,
            config.effective_n_kv_heads * head_dim,
        )
        self.attn_proj = nnx.Linear(
            config.d_model,
            sum(self.fused_dims),
            use_bias=config.include_bias,
            rngs=rngs,
        )

        # Feed-forward projections
        self.ff_proj = nnx.Linear(
            config.d_model,
            self.hidden_size,
            use_bias=config.include_bias,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jnp.ndarray,
        attention_bias: jnp.ndarray | None = None,
        layer_past: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        """Forward pass through the sequential OLMo block.

        Args:
            x: the input tensor
            attention_bias: the attention bias tensor
            layer_past: the past layer tensor

        Returns:
            the output tensor.
        """
        qkv = self.attn_proj(self.attn_norm(x))

        if self.config.clip_qkv is not None:
            qkv = jnp.clip(qkv, -self.config.clip_qkv, self.config.clip_qkv)

        head_size = self.config.d_model // self.config.n_heads
        fused_dims = (
            self.config.d_model,
            self.config.effective_n_kv_heads * head_size,
            self.config.effective_n_kv_heads * head_size,
        )
        split_indices = [sum(fused_dims[:i]) for i in range(1, len(fused_dims))]

        q, k, v = jnp.split(qkv, split_indices, axis=-1)

        attention = self.attention(q, k, v, attention_bias, layer_past=layer_past)

        x = x + self.dropout(attention)

        pre_x = x
        x = self.ff_norm(x)
        x = self.ff_proj(x)
        x = self.activation(x)
        x = self.ff_out(x)
        x = self.dropout(x)
        x = x + pre_x

        return x
