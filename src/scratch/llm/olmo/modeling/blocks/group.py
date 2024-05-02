"""Block groups."""

from collections.abc import Iterable

import torch
from torch import nn

from scratch.llm.olmo.modeling.config import OLMoConfig


class BlockGroup(nn.ModuleList):
    """Group of blocks.

    This is used for distributed training.
    """

    def __init__(
        self,
        config: OLMoConfig,
        layer_offset: int,
        modules: Iterable[nn.Module] | None = None,
    ):
        """Initialize the block group.

        Args:
            config: Configuration object.
            layer_offset: Offset for the layer index.
            modules: List of modules to add to the block group.
        """
        super().__init__(modules)
        self.config = config
        self.layer_offset = layer_offset

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: torch.FloatTensor | None = None,
        layers_past: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        *,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        """Forward pass through the block group.

        Args:
          x: Input tensor.
          attention_bias: Attention bias tensor.
          layers_past: List of past layers for each block.
          use_cache: Whether to use cache for attention.

        Returns:
          Tuple of the output tensor and the attention key values.
        """
        attn_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = (
            [] if use_cache else None
        )
        for block_idx, block in enumerate(self):
            layer_past = None if layers_past is None else layers_past[block_idx]
            block_idx += self.layer_offset
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
        return x, attn_key_values

    def reset_parameters(self):
        """Reset the parameters of the block group."""
        for block in self:
            block.reset_parameters()
