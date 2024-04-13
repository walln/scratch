"""Layer initializations."""

from enum import Enum

import torch
from torch import nn

from scratch.llm.olmo.modeling.model import OLMoBlockConfig


class ModuleType(Enum):
    """Type of module to initialize."""

    in_module = "in"
    out_module = "out"
    emb = "emb"
    final_out = "final_out"


def init_weights(
    config: OLMoBlockConfig,
    module: nn.Linear | nn.Embedding,
    d: int | None = None,
    layer_id: int | None = None,
    std_factor: float = 1.0,
    type_of_module: ModuleType | None = None,
):
    """Initialize weights of a linear or embedding layer.

    Args:
      config: Model configuration.
      module: Linear or embedding layer to initialize.
      d: Dimension of the layer.
      layer_id: ID of the layer.
      std_factor: Factor to multiply the standard deviation by.
      type_of_module: Type of module to initialize.
    """
    raise NotImplementedError()


def _non_meta_init_device(config: OLMoBlockConfig) -> torch.device:
    if config.init_device is not None and config.init_device != "meta":
        return torch.device(config.init_device)
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
