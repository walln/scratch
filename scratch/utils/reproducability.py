import torch


def set_random_seed(seed: int = 1):
    torch.manual_seed(seed)
