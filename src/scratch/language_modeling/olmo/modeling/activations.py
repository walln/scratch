"""Activation functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU activation function.

    SwiGLU activation function is defined as:
     x * sigmoid(beta * x) + (1 - sigmoid(beta * x)) * (Wx + b)

    This linear gating function is smoother than ReLU and is non-monotonic.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU activation function."""
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

    @property
    def output_multiplier(self) -> float:
        """Return the output multiplier for the activation function."""
        return 0.5
