"""Sequential Block."""

import torch
from torch import nn

from scratch.llm.olmo.modeling.blocks.base import OLMoBlock
from scratch.llm.olmo.modeling.buffer_cache import BufferCache
from scratch.llm.olmo.modeling.config import OLMoConfig
from scratch.llm.olmo.modeling.initializations import ModuleType, init_weights
from scratch.llm.olmo.modeling.layer_norm import create_layer_norm


class SequentialOLMoBlock(OLMoBlock):
    """Standard OLMo transformer block.

    Computes: ``MLP(LN(x + Attention(LN(x))))`` and adds a residual connection.
    """

    def __init__(self, layer_id: int, config: OLMoConfig, cache: BufferCache):
        """Initialize the sequential OLMo block.

        Args:
            layer_id: the ID of the layer
            config: the model configuration
            cache: the buffer cache
        """
        super().__init__(layer_id, config, cache)

        # Create layer norms
        self.attn_norm = create_layer_norm(config)
        self.ff_norm = create_layer_norm(config)

        # Attention projections
        head_dim = config.d_model // config.n_heads
        self.fused_dims = (
            config.d_model,
            config.effective_n_kv_heads * head_dim,
            config.effective_n_kv_heads * head_dim,
        )
        self.attn_proj = nn.Linear(
            config.d_model,
            self.hidden_size,
            bias=config.include_bias,
            device=config.init_device,
        )

        # Feed-forward projections
        self.ff_proj = nn.Linear(
            config.d_model,
            self.hidden_size,
            bias=config.include_bias,
            device=config.init_device,
        )

    def reset_parameters(self):
        """Reset the parameters of the model.

        This includes the attention and feedforward projections, as well as the
        layer norms.
        """
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()

        # NOTE: the standard deviation for these weights does not depend on the layer.
        init_weights(
            self.config,
            self.attn_proj,
            d=self.config.d_model,
            layer_id=None,
            type_of_module=ModuleType.in_module,
        )
        init_weights(
            self.config,
            self.ff_proj,
            d=self.config.d_model,
            layer_id=None,
            type_of_module=ModuleType.in_module,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
        layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache=False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Forward pass through the sequential OLMo block."""
        # Get Q,K,V projections
        # if using MHA, the shape is q,k,v: (B, S, D)
        # for MQA, the shape is q: (B, S, D) k,v: (B, S, D // n_heads)

        qkv = self.attn_proj(self.attn_norm(x))

        if self.config.clip_qkv is not None:
            qkv.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        q, k, v = qkv.split(self.fused_dims, dim=-1)

        # Compute attention
        attention, cache = self.attention(
            q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache
        )

        # Add and normalize

        # shape: (B, T, C)
        x = x + self.dropout(attention)

        # Add feedforward projection
        # shape: (B, S, D)
        pre_x = x
        x = self.ff_norm(x)
        x = self.ff_proj(x)
        x = self.activation(x)
        x = self.ff_out(x)
        x = self.dropout(x)
        x = x + pre_x

        return x, cache
