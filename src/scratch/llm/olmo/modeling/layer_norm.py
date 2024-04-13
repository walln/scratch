"""Layer normalizations."""

from torch import nn

from scratch.llm.olmo.modeling.model import OLMoBlockConfig


def create_layer_norm_base(
    block_config: OLMoBlockConfig,
    *,
    size: int | None = None,
    elementwise_affine=False,
    eps=1e-5,
):
    """Create a base layer normalization layer.

    Args:
      block_config: Block configuration.
      size: Size of the layer normalization layer.
      elementwise_affine: Whether to use elementwise affine.
      eps: Epsilon value.
    """
    raise NotImplementedError()


def create_layer_norm(
    block_config: OLMoBlockConfig,
    size: int | None = None,
    low_precision=False,
    elementwise_affine=False,
    eps=1e-5,
) -> nn.LayerNorm:
    """Create a layer normalization layer.

    Args:
      block_config: Block configuration.
      size: Size of the layer normalization layer.
      low_precision: Whether to use low precision.
      elementwise_affine: Whether to use elementwise affine.
      eps: Epsilon value.
    """
    raise NotImplementedError()
