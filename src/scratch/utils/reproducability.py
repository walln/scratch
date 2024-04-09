"""Reproducability utilities."""

import torch


def set_random_seed(seed: int = 1):
    """Set the random seed."""
    torch.manual_seed(seed)
