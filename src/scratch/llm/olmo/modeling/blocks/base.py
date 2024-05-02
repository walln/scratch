"""Base OLMo transformer block."""

import torch
import torch.nn.functional as F
from torch import nn

from scratch.llm.olmo.modeling.activations import SwiGLU
from scratch.llm.olmo.modeling.buffer_cache import BufferCache
from scratch.llm.olmo.modeling.config import OLMoConfig
from scratch.llm.olmo.modeling.initializations import ModuleType, init_weights
from scratch.llm.olmo.modeling.rotary_embedding import RotaryEmbedding


class OLMoBlock(nn.Module):
    """Base OLMo transformer block."""

    def __init__(self, layer_id: int, config: OLMoConfig, cache: BufferCache):
        """Initialize the OLMo block.

        Args:
            layer_id: the ID of the layer
            config: the model configuration
            cache: the buffer cache
        """
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = config.mlp_ratio * config.d_model

        # Ensure the model dimensions are valid
        assert config.d_model % config.n_heads == 0

        self.dropout = nn.Dropout(config.residual_dropout)

        # Make sure QKV clip coefficient is positive, otherwise it's not well-defined.
        if config.clip_qkv is not None:
            assert config.clip_qkv > 0

        self.activation = SwiGLU()
        assert (self.activation.output_multiplier * self.hidden_size) % 1 == 0

        # Compute attention projections
        self.attention_out = nn.Linear(
            config.d_model,
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )

        # Feed-forward network
        self.ff_out = nn.Linear(
            int(self.act.output_multiplier * self.hidden_size),
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )
        self.ff_out._is_residual = True  # type: ignore

        if config.use_rope:
            self.rotary_embedding = RotaryEmbedding(config, cache)

    def reset_parameters(self):
        """Reset the parameters of the model.

        This includes the attention and feedforward projections, as well as the
        layer norms.
        """
        if self.k_norm is not None:
            self.k_norm.reset_parameters()
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        init_weights(
            self.config,
            self.attn_out,
            d=self.config.d_model,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )
        init_weights(
            self.config,
            self.ff_out,
            d=self.ff_out.in_features,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        dropout: float = 0.0,
        is_causal=False,
    ) -> torch.Tensor:
        """Computes scaled dot-product attention.

        Args:
            q: the query tensor
            k: the key tensor
            v: the value tensor
            attention_mask: the attention mask tensor
            dropout: the dropout rate
            is_causal: whether the attention is causal
        """
        # torch's sdpa doesn't support GQA, so we're doing this
        assert k.size(1) == v.size(1)
        num_kv_heads = k.size(1)
        num_q_heads = q.size(1)
        if num_q_heads != num_kv_heads:
            assert num_q_heads % num_kv_heads == 0
            k = k.repeat_interleave(
                num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads
            )
            v = v.repeat_interleave(
                num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads
            )

        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=dropout,
            is_causal=is_causal,
        )

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
        layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache=False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Compute attention scores.

        Args:
            q: the query tensor
            k: the key tensor
            v: the value tensor
            attention_bias: the attention bias tensor
            layer_past: the past layer
            use_cache: whether to use cache
        """
        B, T, C = q.size()  # batch size, sequence length, d_model
        dtype = k.dtype

        # apply layer norm if needed
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        # Move head forward to be next to the batch dim.
        # shape: (B, nh, T, hs)
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        k = k.view(
            B, T, self.config.effective_n_kv_heads, C // self.config.n_heads
        ).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        v = v.view(
            B, T, self.config.effective_n_kv_heads, C // self.config.n_heads
        ).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        present = (k, v) if use_cache else None
        query_len, key_len = q.shape[-2], k.shape[-2]

        if self.config.use_rope:
            # Apply rotary embeddings
            q, k = self.rotary_embedding(q, k)

        if attention_bias is not None:
            # resize/cast attention bias - this is and AMP issue
            attention_bias = self._cast_attn_bias(
                attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
            )

        # Compute attention scores
        # shape: (B, nh, T, hs)
        attention = self._scaled_dot_product_attention(
            q,
            k,
            v,
            attention_mask=attention_bias,
            dropout=0.0 if not self.training else self.config.attention_dropout,
            is_causal=attention_bias is None,
        )

        # put the outputs of the heads together again
        attention = attention.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        return self.attention_out(attention), present

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
        layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache=False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Forward pass through the OLMo block."""
        raise NotImplementedError()
