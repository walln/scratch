"""OLMo model implementation."""

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from scratch.language_modeling.olmo.modeling.blocks.group import BlockGroup
from scratch.language_modeling.olmo.modeling.blocks.sequential import (
    SequentialOLMoBlock,
)
from scratch.language_modeling.olmo.modeling.buffer_cache import BufferCache
from scratch.language_modeling.olmo.modeling.config import OLMoConfig
from scratch.language_modeling.olmo.modeling.dropout import Dropout
from scratch.language_modeling.olmo.modeling.initializations import (
    ModuleType,
    _non_meta_init_device,
    init_weights,
)
from scratch.language_modeling.olmo.modeling.layer_norm import create_layer_norm
from scratch.language_modeling.olmo.utils.attention import get_causal_attention_bias
from scratch.language_modeling.olmo.utils.numerical_stability import ensure_finite

base_config = OLMoConfig()


class OLMoForwardResult(NamedTuple):
    """Forward pass result for OLMo model.

    Attributes:
        logits: A tensor of shape (batch_size, seq_len, vocab_size) representing the
            log probs for the next token before normalization through log softmax.
        attn_key_values: Attention keys and values from each block.
        hidden_states: Hidden states from each block.
    """

    logits: torch.Tensor
    attn_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None
    hidden_states: tuple[torch.Tensor] | None


class OLMo(nn.Module):
    """OLMo model."""

    def __init__(self, config: OLMoConfig, *, init_params: bool = True):
        """Initialize the model.

        Args:
            config: The model configuration.
            init_params: Whether to initialize the model parameters.
        """
        super().__init__()
        self.config = config

        self.__cache = BufferCache()

        if (
            self.config.embedding_size is not None
            and self.config.embedding_size != self.config.vocab_size
        ):
            if self.config.embedding_size < self.config.vocab_size:
                raise ValueError(
                    "embedding_size must be greater than or equal to vocab_size"
                )
        elif self.config.embedding_size % 128 != 0:
            # for throughput reasons, we want the embedding size to be a multiple of 128
            raise ValueError("embedding_size must be a multiple of 128")

        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # slows down flash sdp

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(
                    config.embedding_size or config.vocab_size,
                    config.d_model,
                    device=config.init_device,
                ),
                "emb_drop": Dropout(config.embedding_dropout),
                # "ln_f": LayerNorm.build(config),
                "ln_f": create_layer_norm(config),
            }
        )

        blocks = [
            SequentialOLMoBlock(i, config, self.__cache) for i in range(config.n_layers)
        ]
        if self.config.block_group_size > 1:
            block_groups = [
                BlockGroup(config, i, blocks[i : i + config.block_group_size])
                for i in range(0, config.n_layers, config.block_group_size)
            ]
            self.transformer.update({"block_groups": nn.ModuleList(block_groups)})
        else:
            self.transformer.update({"blocks": nn.ModuleList(blocks)})

        if not config.weight_tying:
            self.transformer.update(
                {
                    "ff_out": nn.Linear(
                        config.d_model,
                        config.embedding_size or config.vocab_size,
                        bias=config.include_bias,
                        device=config.init_device,
                    )
                }
            )

        if init_params and self.config.init_device != "meta":
            self.reset_parameters()

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        device: torch.device = self.transformer.wte.weight.device  # type: ignore
        if device.type == "meta":
            return _non_meta_init_device(self.config)
        else:
            return device

    @property
    def num_params(self, *, include_embedding: bool = True) -> int:
        """Get the total number of parameters.

        Args:
            include_embedding: Whether to include the embedding parameters.

        Returns:
            The total number of parameters.
        """
        params = (np for np in self.named_parameters())
        if not include_embedding:
            params = filter(  # type: ignore
                lambda np: ".wte." not in np[0] and ".wpe." not in np[0],
                params,
            )
        return sum(p.numel() for _, p in params)

    def reset_parameters(self):
        """Reset the parameters of the model.

        This includes the embeddings, layer norms, and output weights.
        """
        # TODO: better logging
        # log.info("Initializing model parameters...")

        # Top-level embeddings / linear layers.
        init_weights(
            self.config,
            self.transformer.wte,  # type: ignore
            std_factor=(0.5 * math.sqrt(self.config.d_model))
            if self.config.scale_logits
            else 1.0,
            type_of_module=ModuleType.emb,
        )
        if hasattr(self.transformer, "wpe"):
            init_weights(
                self.config, self.transformer.wpe, type_of_module=ModuleType.emb
            )  # type: ignore

        # Top-level layer norm.
        self.transformer.ln_f.reset_parameters()  # type: ignore

        # Output weights.
        if hasattr(self.transformer, "ff_out"):
            init_weights(
                self.config,
                self.transformer.ff_out,
                type_of_module=ModuleType.final_out,
            )  # type: ignore

        # Let the blocks handle themselves.
        if self.config.block_group_size == 1:
            for block in self.transformer.blocks:
                block.reset_parameters()
        else:
            for block_group in self.transformer.block_groups:
                block_group.reset_parameters()

    def forward(
        self,
        input_ids: torch.Tensor,
        input_embeddings: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        past_key_values: list[torch.Tensor] | None = None,
        *,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: bool = False,
    ) -> OLMoForwardResult:
        """Forward pass through the OLMo model.

        Args:
            input_ids: The input token IDs. (batch_size, seq_len)
            input_embeddings: The input embeddings. (batch_size, seq_len, d_model)
                if provided, this will be used instead of the embeddings from the
                token IDs.
            attention_mask: The attention mask. (batch_size, seq_len)
                Tensor indicating which input ids should be attended to.
            attention_bias: The attention bias. (batch_size, 1, seq_len, seq_len)
                Tensor indicating the introduction of causal bias.
            past_key_values: The past key values. Pre-computed key and values for each
                attention layer. (batch_size, n_layers, 2, n_heads, seq_len, head_dim)
            use_cache: Whether to use the cache to return K,V values for each block.
            last_logits_only: Whether to return only the last logits, for fast decoding.
            output_hidden_states: Whether to output hidden states.

        Returns:
            The logits, attention key values, and hidden states.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )

        if past_key_values:
            assert len(past_key_values) == self.config.n_layers

        batch_size, seq_len = (
            input_ids.size()
            if input_embeddings is None
            else input_embeddings.size()[:2]
        )
        past_len = 0 if past_key_values is None else past_key_values[0][0].size(-2)

        # Compute embeddings (batch_size, seq_len, d_model)
        x = (
            self.transformer.wte(input_ids)
            if input_embeddings is None
            else input_embeddings
        )

        x = self.transformer.emb_drop(x)

        # Attention masking
        if attention_mask is not None:
            # (batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[
                :, None, None, :
            ]
            attention_mask = (1.0 - attention_mask) * torch.finfo(
                attention_mask.dtype
            ).min

        # Merge mask
        if (
            attention_bias is not None
            or attention_mask is not None
            or past_key_values is not None
        ):
            if attention_bias is None:
                attention_bias = get_causal_attention_bias(
                    self.__cache, past_len + seq_len, x.device
                )
            elif attention_bias.dtype in (torch.uint8, torch.bool):
                attention_bias = attention_bias.to(dtype=torch.float)
                attention_bias.masked_fill_(
                    attention_bias == 0.0, torch.finfo(attention_bias.dtype).min
                )

            # Fix shape and dtype
            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            elif past_key_values is not None:
                mask_len = past_key_values[0][0].shape[-2] + seq_len

            attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(  # type: ignore cant be none here
                dtype=torch.float
            )

            # add in bias
            if attention_bias is not None:
                attention_bias = attention_bias + attention_mask
                ensure_finite(attention_bias, check_neg_inf=True, check_pos_inf=False)

        attn_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = (
            [] if use_cache else None
        )

        all_hidden_states = []

        if self.config.block_group_size == 1:
            for block_idx, block in enumerate(self.transformer.blocks):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layer_past = (
                    None if past_key_values is None else past_key_values[block_idx]
                )

                # shape: (batch_size, seq_len, d_model)
                x, cache = block(
                    x,
                    attention_bias=attention_bias,
                    layer_past=layer_past,
                    use_cache=use_cache,
                )

                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.append(cache)
        else:
            for group_idx, block_group in enumerate(self.transformer.block_groups):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layers_past = (
                    None
                    if past_key_values is None
                    else past_key_values[
                        group_idx * self.config.block_group_size : (group_idx + 1)
                        * self.config.block_group_size
                    ]
                )
                x, cache = block_group(
                    x,
                    attention_bias=attention_bias,
                    layers_past=layers_past,
                    use_cache=use_cache,
                )
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.extend(cache)

        if last_logits_only:
            x = x[:, -1, :].unsqueeze(1)  # (batch_size, 1, d_model)

        # layer norm
        # (batch_size, seq_len | 1, d_model)
        x = self.transformer.ln_f(x)
        if output_hidden_states:
            all_hidden_states.append(x)

        # (batch_size, seq_len | 1, vocab_size)
        if self.config.weight_tying:
            logits = F.linear(x, self.transformer.wte.weight, None)
        else:
            logits = self.transformer.ff_out(x)

        if self.config.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.d_model))

        return OLMoForwardResult(
            logits=logits,
            attn_key_values=attn_key_values,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
        )
