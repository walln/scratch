"""Shared dropout layer configuration."""

import torch
import torch.nn.functional as F
from torch import nn


class Dropout(nn.Dropout):
    """Dropout layer with common defaults."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass through the dropout layer.

        Args:
          input: Input tensor.

        Returns:
          Output tensor.
        """
        if self.p == 0.0:
            return input
        else:
            return F.dropout(input, self.p, self.training, self.inplace)
