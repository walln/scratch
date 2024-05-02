"""Layer normalizations."""

from abc import abstractmethod

import torch
import torch.nn.functional as F
from torch import nn

from scratch.llm.olmo.modeling.config import OLMoConfig


class LayerNormBase(nn.Module):
    """Base class for layer normalization layers."""

    def __init__(
        self,
        config: OLMoConfig,
        *,
        size: int | None = None,
        elementwise_affine: bool | None = True,
        eps: float = 1e-05,
    ):
        """Initialize the layer normalization layer.

        Args:
          config: Block configuration.
          size: Size of the layer normalization layer.
          elementwise_affine: Whether to use an affine transformation.
          eps: Epsilon value for numerical stability.
        """
        super().__init__()
        self.config = config
        self.eps = eps
        self.normalized_shape = (size or config.d_model,)
        if elementwise_affine or (elementwise_affine is None):
            self.weight = nn.Parameter(
                torch.ones(self.normalized_shape, device=config.init_device)
            )
            use_bias = self.config.include_bias
            if use_bias:
                self.bias = nn.Parameter(
                    torch.zeros(self.normalized_shape, device=config.init_device)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)
            self.register_parameter("weight", None)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the layer normalization layer.

        Args:
          x: Input tensor.

        Returns:
          Output tensor.
        """
        raise NotImplementedError

    def _cast_if_autocast_enabled(
        self, tensor: torch.Tensor, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        # is_autocast_enabled() only checks for CUDA autocast
        # is_autocast_cpu_enabled() checks for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if tensor.device.type == "cuda" and torch.is_autocast_enabled():
            return tensor.to(
                dtype=dtype if dtype is not None else torch.get_autocast_gpu_dtype()
            )
        elif tensor.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            return tensor.to(
                dtype=dtype if dtype is not None else torch.get_autocast_cpu_dtype()
            )
        else:
            return tensor

    def reset_parameters(self):
        """Reset the layer normalization layer parameters."""
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class LayerNorm(LayerNormBase):
    """The default LayerNorm implementation which can run in low precision."""

    def __init__(
        self,
        config: OLMoConfig,
        size: int | None = None,
        *,
        low_precision: bool = False,
        elementwise_affine: bool | None = None,
        eps: float = 1e-05,
    ):
        """Initialize the layer normalization layer.

        Args:
          config: Block configuration.
          size: Size of the layer normalization layer.
          low_precision: Whether to run the layer normalization in low precision.
          elementwise_affine: Whether to use an affine transformation.
          eps: Epsilon value for numerical stability.
        """
        super().__init__(
            config, size=size, elementwise_affine=elementwise_affine, eps=eps
        )
        self.low_precision = low_precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the layer normalization layer.

        Args:
          x: Input tensor.

        Returns:
          Output tensor.
        """
        if self.low_precision:
            module_device = x.device
            downcast_x = self._cast_if_autocast_enabled(x)
            downcast_weight = (
                self._cast_if_autocast_enabled(self.weight)
                if self.weight is not None
                else self.weight
            )
            downcast_bias = (
                self._cast_if_autocast_enabled(self.bias)
                if self.bias is not None
                else self.bias
            )
            with torch.autocast(enabled=False, device_type=module_device.type):
                return F.layer_norm(
                    downcast_x,
                    self.normalized_shape,
                    weight=downcast_weight,
                    bias=downcast_bias,
                    eps=self.eps,
                )
        else:
            return F.layer_norm(
                x,
                self.normalized_shape,
                weight=self.weight,
                bias=self.bias,
                eps=self.eps,
            )


def create_layer_norm(
    config: OLMoConfig, size: int | None = None, **kwargs
) -> LayerNorm:
    """Create a layer normalization layer.

    Args:
      config: Block configuration.
      size: Size of the layer normalization layer.
      kwargs: Additional keyword arguments.
    """
    return LayerNorm(config, size=size, low_precision=False, **kwargs)
