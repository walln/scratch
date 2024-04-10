"""Test multi-head attention layer."""

import pytest
import torch

from scratch.deep_learning.layers.attention import MultiHeadAttention


def test_initialization():
    """Test the initialization of the MultiHeadAttention layer."""
    embed_dim, d_model, num_heads = 128, 512, 8
    mha = MultiHeadAttention(embed_dim, d_model, num_heads)
    assert mha.d_head == d_model // num_heads
    assert mha.num_heads == num_heads
    assert mha.qkv_projection.in_features == embed_dim
    assert mha.qkv_projection.out_features == 3 * d_model
    assert mha.output_projection.in_features == d_model
    assert mha.output_projection.out_features == d_model


def test_forward_shape():
    """Test the forward method's output shape."""
    batch_size, seq_length, embed_dim = 2, 16, 128
    d_model, num_heads = 512, 8
    x = torch.rand(batch_size, seq_length, embed_dim)
    mask = torch.ones(batch_size, seq_length, seq_length)

    mha = MultiHeadAttention(embed_dim, d_model, num_heads)
    output = mha(x, mask=mask)
    assert output.size() == (batch_size, seq_length, d_model)

    output, attention = mha(x, mask=mask, return_attention=True)
    assert output.size() == (batch_size, seq_length, d_model)
    assert attention.size() == (batch_size, num_heads, seq_length, seq_length)


@pytest.mark.parametrize("return_attention", [True, False])
def test_forward_return_attention(return_attention):
    """Test if the forward method returns the attention tensor when requested."""
    batch_size, seq_length, embed_dim = 2, 16, 128
    d_model, num_heads = 512, 8
    x = torch.rand(batch_size, seq_length, embed_dim)
    mha = MultiHeadAttention(embed_dim, d_model, num_heads)
    output = mha(x, return_attention=return_attention)

    if return_attention:
        assert isinstance(output, tuple)
        assert len(output) == 2
        assert isinstance(output[1], torch.Tensor)
    else:
        assert isinstance(output, torch.Tensor)


def test_attention_values_validity():
    """Test if attention values are within the valid range [0, 1]."""
    batch_size, seq_length, embed_dim = 2, 16, 128
    d_model, num_heads = 512, 8
    x = torch.rand(batch_size, seq_length, embed_dim)
    mask = torch.ones(batch_size, seq_length, seq_length)

    mha = MultiHeadAttention(embed_dim, d_model, num_heads)
    _, attention = mha(x, mask=mask, return_attention=True)

    # Check attention values are within the valid range [0, 1]
    assert torch.all(attention >= 0), "Attention values must be non-negative."
    assert torch.all(
        attention <= 1
    ), "Attention values must be less than or equal to 1."

    # Check if the sum of the attention weights for each query across all keys is 1
    attention_sum = attention.sum(dim=-1)
    assert torch.allclose(
        attention_sum, torch.ones_like(attention_sum)
    ), "Sum of attention weights must be 1."
