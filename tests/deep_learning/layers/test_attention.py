"""Test attention layers."""

import pytest
import torch

from scratch.deep_learning.layers.attention import (
    DotProductAttention,
    ScaledDotProductAttention,
    SelfAttention,
)


@pytest.fixture()
def input_tensor():
    """Fixture to provide a sample input tensor."""
    # Creating a batch of 2 sequences, each of length 3, with an input dimension of 4
    return torch.rand(2, 3, 4)


@pytest.fixture()
def query_tensor():
    """Fixture to provide a sample query tensor."""
    # Creating a batch of 2 sequences, each with an input dimension of 4
    return torch.rand(2, 1, 4)


@pytest.fixture()
def value_tensor():
    """Fixture to provide a sample value tensor."""
    # Creating a batch of 2 sequences, each of length 3, with an input dimension of 4
    return torch.rand(2, 3, 4)


@pytest.fixture()
def self_attention_layer():
    """Fixture to provide a configured SelfAttention layer instance."""
    input_dim = 4  # Match the last dimension of the input_tensor
    return SelfAttention(input_dim)


def test_self_attention_output_shape(self_attention_layer, input_tensor):
    """Test if the SelfAttention layer returns the correct output shape."""
    output, attention = self_attention_layer.forward(input_tensor)
    assert (
        output.shape == input_tensor.shape
    ), "Output tensor does not match the input tensor shape."
    assert attention.shape == (
        input_tensor.shape[0],
        input_tensor.shape[1],
        input_tensor.shape[1],
    ), "Attention tensor has an incorrect shape."


def test_self_attention_output_type(self_attention_layer, input_tensor):
    """Test if the SelfAttention layer returns a tensor."""
    output, attention = self_attention_layer(input_tensor)
    assert isinstance(output, torch.Tensor), "Output is not a PyTorch tensor."
    assert isinstance(attention, torch.Tensor), "Attention is not a PyTorch tensor."


def test_self_attention_scores_range(self_attention_layer, input_tensor):
    """Test if the attention scores are within the expected range [0, 1]."""
    _, attention_scores = self_attention_layer.forward(input_tensor)
    assert torch.all(attention_scores >= 0), "Attention scores are negative."
    assert torch.all(attention_scores <= 1), "Attention scores are greater than 1."
    assert torch.allclose(
        attention_scores.sum(dim=-1), torch.ones_like(attention_scores.sum(dim=-1))
    ), "Attention scores do not sum to 1."


@pytest.fixture()
def dot_product_attention_layer():
    """Fixture to provide a configured DotProductAttention layer instance."""
    return DotProductAttention()


def test_dot_product_attention_output_shapes(
    dot_product_attention_layer, query_tensor, value_tensor
):
    """Test if the DotProductAttention layer returns the correct output shapes."""
    context, attention = dot_product_attention_layer(query_tensor, value_tensor)
    assert context.shape == (
        query_tensor.size(0),
        query_tensor.size(1),
        value_tensor.size(2),
    ), "Context tensor shape is incorrect."
    assert attention.shape == (
        query_tensor.size(0),
        query_tensor.size(1),
        value_tensor.size(1),
    ), "Attention tensor shape is incorrect."


def test_dot_product_attention_output_types(
    dot_product_attention_layer, query_tensor, value_tensor
):
    """Test if the DotProductAttention layer returns tensors."""
    context, attention = dot_product_attention_layer(query_tensor, value_tensor)
    assert isinstance(context, torch.Tensor), "Context is not a PyTorch tensor."
    assert isinstance(attention, torch.Tensor), "Attention is not a PyTorch tensor."


def test_dot_product_attention_scores_range(
    dot_product_attention_layer, query_tensor, value_tensor
):
    """Test if the attention scores are within the expected range [0, 1]."""
    _, attention_scores = dot_product_attention_layer(query_tensor, value_tensor)

    assert torch.all(attention_scores >= 0), "Attention scores are negative."
    assert torch.all(attention_scores <= 1), "Attention scores are greater than 1."
    assert torch.allclose(
        attention_scores.sum(dim=-1), torch.ones_like(attention_scores.sum(dim=-1))
    ), "Attention scores do not sum to 1."


@pytest.fixture()
def sdpa_query_tensor():
    """Provides a sample query tensor."""
    # Creating a batch of 2 sequences, each with an input dimension of 5
    return torch.rand(2, 1, 5)


@pytest.fixture()
def sdpa_key_value_tensor():
    """Provides a sample key/value tensor."""
    # Creating a batch of 2 sequences, each of length 3, with an input dimension of 5
    return torch.rand(2, 3, 5)


@pytest.fixture()
def sdpa_mask_tensor():
    """Provides a sample mask tensor that can be broadcasted to the scores shape."""
    batch_size = 2
    seq_length = 3  # Assuming 'seq_length' matches the 'num_keys'
    # Initialize mask with False (positions are not masked by default)
    mask = torch.zeros(batch_size, 1, seq_length, dtype=torch.bool)
    # Example: Masking out the first position in each sequence
    mask[:, :, 0] = True  # Now, the first position is masked
    return mask


@pytest.fixture()
def scaled_dot_product_attention_layer():
    """Provides a configured ScaledDotProductAttention layer instance."""
    d_head = 5  # This should match the last dimension of the query/key/value tensors
    return ScaledDotProductAttention(d_head)


def test_scaled_dot_product_attention_output_shapes(
    scaled_dot_product_attention_layer, sdpa_query_tensor, sdpa_key_value_tensor
):
    """Tests if the outputs have the correct shapes."""
    context, attention = scaled_dot_product_attention_layer(
        sdpa_query_tensor, sdpa_key_value_tensor, sdpa_key_value_tensor
    )
    assert (
        context.shape == sdpa_query_tensor.shape
    ), "Context tensor shape is incorrect."
    assert attention.shape == (
        sdpa_query_tensor.size(0),
        sdpa_query_tensor.size(1),
        sdpa_key_value_tensor.size(1),
    ), "Attention tensor shape is incorrect."


def test_scaled_dot_product_attention_output_types(
    scaled_dot_product_attention_layer, sdpa_query_tensor, sdpa_key_value_tensor
):
    """Tests if the outputs are tensors."""
    context, attention = scaled_dot_product_attention_layer(
        sdpa_query_tensor, sdpa_key_value_tensor, sdpa_key_value_tensor
    )
    assert isinstance(context, torch.Tensor), "Context is not a PyTorch tensor."
    assert isinstance(attention, torch.Tensor), "Attention is not a PyTorch tensor."


def test_scaled_dot_product_attention_scores_range(
    scaled_dot_product_attention_layer, sdpa_query_tensor, sdpa_key_value_tensor
):
    """Tests if the attention scores are within the expected range [0, 1]."""
    _, attention_scores = scaled_dot_product_attention_layer(
        sdpa_query_tensor, sdpa_key_value_tensor, sdpa_key_value_tensor
    )
    assert torch.all(attention_scores >= 0), "Attention scores are negative."
    assert torch.all(attention_scores <= 1), "Attention scores are greater than 1."
    assert torch.allclose(
        attention_scores.sum(dim=-1), torch.ones_like(attention_scores.sum(dim=-1))
    ), "Attention scores do not sum to 1."


def test_scaled_dot_product_attention_with_mask(
    scaled_dot_product_attention_layer,
    sdpa_query_tensor,
    sdpa_key_value_tensor,
    sdpa_mask_tensor,
):
    """Tests if the attention layer effectively ignores masked positions."""
    _, attention_scores = scaled_dot_product_attention_layer(
        sdpa_query_tensor,
        sdpa_key_value_tensor,
        sdpa_key_value_tensor,
        sdpa_mask_tensor,
    )
    # Check if attention scores for masked positions are significantly lower
    masked_attention_scores = attention_scores[
        :, :, 0
    ]  # Assuming first position is masked
    assert torch.all(
        masked_attention_scores < 1e-5
    ), "Masked positions are not effectively ignored."
