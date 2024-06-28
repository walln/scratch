"""Buffer cache as found in the official implementation of OLMo.

It appears this is a patch for strange FSDP behavior.
"""

from collections.abc import MutableMapping

import torch


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    """Cache for attention biases and other things that would be stored as buffers.

    In general it appears the way FSDP handles buffers is not well-defined.
    It doesn't shard them but apparently it does synchronize them across processes,
    sometimes have `-inf` in these biases which might get turned into
    NaNs when they're synchronized due to casting or some other issue.
    """
