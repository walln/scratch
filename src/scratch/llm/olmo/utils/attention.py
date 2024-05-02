"""Attention utilities."""

import torch

from scratch.llm.olmo.modeling.buffer_cache import BufferCache


def causal_attention_bias(seq_len: int, device: torch.device) -> torch.FloatTensor:
    """Create a causal attention bias tensor.

    Args:
      seq_len: Sequence length.
      device: Device to use.

    Returns:
      Causal attention bias tensor.
    """
    att_bias = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.float),
        diagonal=1,
    )
    att_bias.masked_fill_(att_bias == 1, torch.finfo(att_bias.dtype).min)
    return att_bias.view(1, 1, seq_len, seq_len)  # type: ignore


def get_causal_attention_bias(
    cache: BufferCache, seq_len: int, device: torch.device
) -> torch.Tensor:
    """Get the causal attention bias tensor.

    Args:
      cache: Buffer cache.
      seq_len: Sequence length.
      device: Device to use.

    Returns:
      Causal attention bias tensor.
    """
    if (
        causal_bias := cache.get("causal_attention_bias")
    ) is not None and causal_bias.shape[-1] >= seq_len:
        if causal_bias.device != device:
            causal_bias = causal_bias.to(device)
            cache["causal_attention_bias"] = causal_bias
        return causal_bias
    with torch.autocast(device.type, enabled=False):
        causal_bias = causal_attention_bias(seq_len, device)
    cache["causal_attention_bias"] = causal_bias
    return causal_bias
