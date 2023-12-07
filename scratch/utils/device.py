import torch
from torch.backends.mps import is_available as mps_is_available
from torch.cuda import is_available as cuda_is_available


def get_device():
    """
    Returns the device to use for pytorch backend.
    If cuda is available, returns cuda device.
    if mps is available, returns mps device.
    Otherwise, returns cpu device as no other devices have
    been tested for these implementations.
    """
    if cuda_is_available():
        return torch.device("cuda")
    if mps_is_available():
        return torch.device("mps")
    return torch.device("cpu")
