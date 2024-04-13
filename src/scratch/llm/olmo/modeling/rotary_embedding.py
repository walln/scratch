"""Rotary Embedding."""

import torch
from torch import einsum, nn

from scratch.llm.olmo.modeling.buffer_cache import BufferCache
from scratch.llm.olmo.modeling.initializations import _non_meta_init_device
from scratch.llm.olmo.modeling.model import OLMoBlockConfig


class RotaryEmbedding(nn.Module):
    """[Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864)."""

    def __init__(self, config: OLMoBlockConfig, cache: BufferCache):
        """Initialize RotaryEmbedding.

        Args:
          config: Block configuration.
          cache: Buffer cache.
        """
        super().__init__()
        self.config = config
        self.__cache = cache
        # Warm up cache.
        self.get_rotary_embedding(
            config.max_sequence_length, _non_meta_init_device(config)
        )

    def get_rotary_embedding(
        self, seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get rotary positional embeddings.

        Args:
          seq_len: Sequence length.
          device: Device to use.

        Returns: Sine and cosine positional embeddings.
        """
        if (
            (pos_sin := self.__cache.get("rope_pos_sin")) is not None
            and (pos_cos := self.__cache.get("rope_pos_cos")) is not None
            and pos_sin.shape[-2] >= seq_len
            and pos_cos.shape[-2] >= seq_len
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                self.__cache["rope_pos_sin"] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                self.__cache["rope_pos_cos"] = pos_cos
            return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]

        with torch.autocast(device.type, enabled=False):
            dim = self.config.d_model // self.config.n_heads
            inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim)
            )
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = einsum("i , j -> i j", seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin, pos_cos = (
                positions.sin()[None, None, :, :],
                positions.cos()[None, None, :, :],
            )
        self.__cache["rope_pos_sin"] = pos_sin
        self.__cache["rope_pos_cos"] = pos_cos
        return pos_sin, pos_cos

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate tensor by 90 degrees.

        Args:
          x: Tensor to rotate.

        Returns: Rotated tensor.
        """
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(
        self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Apply rotary positional embeddings.

        Args:
          pos_sin: Sine positional embeddings.
          pos_cos: Cosine positional embeddings.
          t: Tensor to apply embeddings to.

        Returns: Tensor with positional embeddings applied.
        """
        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary positional embeddings to query and key.

        Args:
          q: Query tensor.
          k: Key tensor.

          Returns: Query and key tensors with positional embeddings applied.
        """
        q_, k_ = q.float(), k.float()
        with torch.autocast(q.device.type, enabled=False):
            query_len, key_len = (
                q_.shape[-2],
                k_.shape[-2],
            )  # could be different if layer_past not None
            pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)
            q_ = self.apply_rotary_pos_emb(
                pos_sin[:, :, key_len - query_len : key_len, :],
                pos_cos[:, :, key_len - query_len : key_len, :],
                q_,
            )
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
        return q_.type_as(q), k_.type_as(k)
