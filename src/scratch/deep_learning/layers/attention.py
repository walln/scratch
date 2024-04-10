"""Implementations of attention layers."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SelfAttention(nn.Module):
    """Simple self-attention layer computes attention scores and outputs."""

    def __init__(self, input_dim: int):
        """Initialize self-attention layer.

        Args:
            input_dim: The input dimension.
        """
        super().__init__()
        self.input_dim = input_dim
        self.key = nn.Linear(input_dim, input_dim)
        self.query = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Compute self-attention forward pass.

        Args:
            x: The input tensor.

        Returns:
            The output tensor and the attention calculated.
        """
        # Compute query, key, and value
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        # Compute attention scores
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim**0.5)
        attention = self.softmax(scores)

        # Compute output
        output = torch.bmm(attention, values)
        return output, attention


class DotProductAttention(nn.Module):
    """Dot-product attention layer.

    Computes the dot products of the query with all values and
    applies a softmax function to get the weights.
    """

    def forward(self, query: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        """Compute dot-product attention forward pass.

        Args:
            query: The query tensor.
            value: The value tensor.

        Returns:
            The context tensor and the attention tensor.
        """
        batch_size, input_size = query.size(0), value.size(1)

        scores = torch.bmm(query, value.transpose(1, 2))
        attention = F.softmax(scores.view(-1, input_size), dim=1).view(
            batch_size, -1, input_size
        )
        context = torch.bmm(attention, value)

        return context, attention


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention layer.

    Computes the dot products of the query with all keys, scales the dot products
    by a factor of square root of the key dimension, and applies a softmax function.
    Originally proposed in [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
    """

    def __init__(self, d_head: int):
        """Initialize ScaledDotProductAttention.

        Args:
            d_head: The head dimension.
        """
        super().__init__()
        self.d_head = d_head

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Compute scaled dot-product attention forward pass.

        Args:
            query: The query tensor.
            key: The key tensor.
            value: The value tensor.
            mask: The mask tensor.

        Returns:
            The context tensor and the attention tensor.
        """
        d_k = key.size(-1)

        scores = torch.bmm(query, key.transpose(1, 2)) / torch.sqrt(
            torch.tensor(d_k, dtype=query.dtype).float()
        )

        if mask is not None:
            scores = scores.masked_fill(mask, -float("inf"))

        attention = F.softmax(scores, dim=-1)
        context = torch.bmm(attention, value)

        return context, attention


def expand_mask(mask: Tensor):
    """Expand mask to support different shapes.

    Supports shapes (batch_size, n_heads, seq_length, seq_length)
     If 2D, broadcast over batch_size and n_heads
     If 3D, broadcast over n_heads
     If 4D, leave as is

    Args:
         mask: The input mask tensor.

    Returns:
    The expanded mask tensor.
    """
    assert (
        mask.ndim >= 2
    ), "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention as proposed in [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

    Rather than performing a single attention function with d_model-dimensional keys,
    values, and queries, MHA projects the queries, keys and values h times with
    different, learned linear projections to d_head dimensions. The projections are then
    concatenated and projected again to obtain the final values. This allows the model
    to jointly attend to information from different representation subspaces at
    different positions.
    """

    def __init__(self, input_dim: int, d_model: int = 512, num_heads: int = 8):
        """Initialize MultiHeadAttention.

        Args:
            input_dim: The input dimension.
            d_model: The input dimension.
            num_heads: The number of heads.
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.input_dim = input_dim
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads

        self.qkv_projection = nn.Linear(input_dim, 3 * d_model)
        self.output_projection = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization from paper
        nn.init.xavier_uniform_(self.qkv_projection.weight)
        self.qkv_projection.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output_projection.weight)
        self.output_projection.bias.data.fill_(0)

    def _scaled_dot_product(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None
    ):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention

    def forward(
        self, x: Tensor, mask: Tensor | None = None, *, return_attention: bool = False
    ) -> tuple[Tensor, Tensor] | Tensor:
        """Compute multi-head attention forward pass.

        Args:
            x: The input tensor.
            mask: The mask tensor.
            return_attention: Whether to return the attention tensor.

        Returns:
                The context tensor and the attention tensor.
        """
        batch_size, seq_length, _ = x.size()
        if mask is not None:
            mask = expand_mask(mask)

        qkv = self.qkv_projection(x)

        # create Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.d_head)
        qkv = qkv.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_length, d_head]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = self._scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.d_model)
        output = self.output_projection(values)

        if return_attention:
            return output, attention
        return output
