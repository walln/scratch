"""BERT Implementation.

This file contains the implementation of the BERT (Bidirectional Encoder
Representations from Transformers). BERT is a pre-trained transformer model
that has achieved state-of-the-art performance on a wide array of natural language
processing tasks through its bidirectional training of Transformer encoders.

The implementation is based on the paper "BERT: Pre-training of Deep Bidirectional
Transformers for Language Understanding" by Jacob Devlin, Ming-Wei Chang, Kenton Lee,
and Kristina Toutanova.

Reference:
    Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding.
    arXiv preprint arXiv:1810.04805. https://arxiv.org/abs/1810.04805


BERT is a transformer-based model that is commonly used for many language processing
tasks, such as sequence classification, token classification, and question answering.

Due to the popularity of BERT I have implemented simple versions of the BERT model
for many of these tasks. The implementation is based on the original BERT paper and
comes with variants for sequence classification, token classification, and question
answering tasks.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from flax import nnx


@dataclass
class BertConfig:
    """Configuration for the BERT model.

    Attributes:
        vocab_size: Size of the vocabulary.
        hidden_size: Size of the hidden layers.
        num_hidden_layers: Number of hidden layers.
        num_attention_heads: Number of attention heads.
        intermediate_size: Size of the intermediate layers.
        hidden_dropout_prob: Dropout rate for hidden layers.
        attention_probs_dropout_prob: Dropout rate for attention probabilities.
        max_position_embeddings: Maximum number of positions.
        type_vocab_size: Size of the token type vocabulary.
        layer_norm_eps: Epsilon value for layer normalization.
    """

    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12


class Embeddings(nnx.Module):
    """BERT Embedding Module.

    This module takes input IDs, token type IDs, and position IDs and returns the
    embeddings for the input sequence.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int,
        type_vocab_size: int,
        layer_norm_eps: float = 1e-12,
        *,
        rngs: nnx.Rngs,
    ):
        """Initializes the embedding module.

        Args:
            vocab_size: Size of the vocabulary.
            hidden_size: Size of the hidden layers.
            max_position_embeddings: Maximum number of positions.
            type_vocab_size: Size of the token type vocabulary.
            layer_norm_eps: Epsilon value for layer normalization.
            rngs: Random number generators.
        """
        self.token_embeddings = nnx.Embed(vocab_size, hidden_size, rngs=rngs)
        self.position_embeddings = nnx.Embed(
            max_position_embeddings, hidden_size, rngs=rngs
        )
        self.segment_embeddings = nnx.Embed(type_vocab_size, hidden_size, rngs=rngs)
        self.layer_norm = nnx.LayerNorm(hidden_size, epsilon=layer_norm_eps, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)

    def __call__(
        self,
        input_ids: jnp.ndarray,
        token_type_ids: jnp.ndarray,
        position_ids: jnp.ndarray,
        train=False,
    ):
        """Computes the embeddings for the input sequence.

        Args:
            input_ids: Input token IDs.
            token_type_ids: Token type IDs.
            position_ids: Position IDs.
            train: Whether the model is in training mode. Defaults to False.
        """
        input_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        segment_embeds = self.segment_embeddings(token_type_ids)

        embeddings = input_embeds + position_embeds + segment_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings, deterministic=not train)
        return embeddings


class TransformerBlock(nnx.Module):
    """BERT Transformer Block.

    This module implements a single transformer block that consists of a multi-head
    self-attention layer followed by a feedforward neural network. Each of these layers
    is preceded by a layer normalization and followed by a dropout layer.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
        layer_norm_eps: float,
        *,
        rngs: nnx.Rngs,
    ):
        """Initializes the transformer block.

        Args:
            hidden_size: Size of the hidden layers.
            num_attention_heads: Number of attention heads.
            intermediate_size: Size of the intermediate layers.
            hidden_dropout_prob: Dropout rate for hidden layers.
            attention_probs_dropout_prob: Dropout rate for attention probabilities.
            layer_norm_eps: Epsilon value for layer normalization.
            rngs: Random number generators.
        """
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_attention_heads,
            in_features=hidden_size,
            dropout_rate=attention_probs_dropout_prob,
            decode=False,
            rngs=rngs,
        )
        self.attention_norm = nnx.LayerNorm(
            hidden_size, epsilon=layer_norm_eps, rngs=rngs
        )
        self.attention_dropout = nnx.Dropout(rate=hidden_dropout_prob, rngs=rngs)

        self.intermediate_dense = nnx.Linear(hidden_size, intermediate_size, rngs=rngs)
        self.output_dense = nnx.Linear(intermediate_size, hidden_size, rngs=rngs)
        self.output_norm = nnx.LayerNorm(hidden_size, epsilon=layer_norm_eps, rngs=rngs)
        self.output_dropout = nnx.Dropout(rate=hidden_dropout_prob, rngs=rngs)

    def __call__(
        self, hidden_states: jnp.ndarray, attention_mask: jnp.ndarray, train=False
    ):
        """Forward pass of the transformer block.

        Args:
            hidden_states: Input array.
            attention_mask: Attention mask.
            train: Whether the model is in training mode. Defaults to False.
        """
        attention_output = self.attention(
            hidden_states,
            hidden_states,
            hidden_states,
            mask=attention_mask,
            deterministic=not train,
        )
        attention_output = self.attention_dropout(
            attention_output, deterministic=not train
        )
        attention_output = self.attention_norm(hidden_states + attention_output)

        intermediate_output = nnx.gelu(self.intermediate_dense(attention_output))
        layer_output = self.output_dense(intermediate_output)
        layer_output = self.output_dropout(layer_output, deterministic=not train)
        layer_output = self.output_norm(attention_output + layer_output)

        return layer_output


class Encoder(nnx.Module):
    """BERT Encoder.

    This module implements the BERT encoder that consists of multiple transformer
    blocks. The encoder takes the input embeddings and applies the transformer blocks
    to the input sequence.
    """

    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
        layer_norm_eps: float = 1e-12,
        *,
        rngs: nnx.Rngs,
    ):
        """Initializes the BERT encoder.

        Args:
            num_hidden_layers: Number of hidden layers.
            hidden_size: Size of the hidden layers.
            num_attention_heads: Number of attention heads.
            intermediate_size: Size of the intermediate layers.
            hidden_dropout_prob: Dropout rate for hidden layers.
            attention_probs_dropout_prob: Dropout rate for attention probabilities.
            layer_norm_eps: Epsilon value for layer normalization.
            rngs: Random number generators.
        """
        self.layers = [
            TransformerBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                layer_norm_eps=layer_norm_eps,
                rngs=rngs,
            )
            for _ in range(num_hidden_layers)
        ]

    def __call__(
        self, hidden_states: jnp.ndarray, attention_mask: jnp.ndarray, train=False
    ):
        """Forward pass of the BERT encoder.

        Args:
            hidden_states: Input array.
            attention_mask: Attention mask.
            train: Whether the model is in training mode. Defaults to False.
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, train)
        return hidden_states


class BertModel(nnx.Module):
    """BERT Model.

    This module implements the BERT model that consists of basic model and a pooler
    prior to the task adapters.
    """

    def __init__(self, config: BertConfig, *, rngs: nnx.Rngs):
        """Initializes the BERT model.

        Args:
            config: Configuration for the BERT model.
            rngs: Random number generators.
        """
        self.embeddings = Embeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_position_embeddings,
            type_vocab_size=config.type_vocab_size,
            layer_norm_eps=config.layer_norm_eps,
            rngs=rngs,
        )
        self.encoder = Encoder(
            num_hidden_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            layer_norm_eps=config.layer_norm_eps,
            rngs=rngs,
        )
        self.pooler = nnx.Linear(config.hidden_size, config.hidden_size, rngs=rngs)

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray,
        token_type_ids: jnp.ndarray | None = None,
        position_ids: jnp.ndarray | None = None,
        train=False,
    ):
        """Forward pass of the BERT model.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            token_type_ids: Token type IDs.
            position_ids: Position IDs.
            train: Whether the model is in training mode. Defaults to False.
        """
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)
        if position_ids is None:
            position_ids = jnp.arange(input_ids.shape[1])

        embedding_output = self.embeddings(
            input_ids, token_type_ids, position_ids, train
        )
        encoder_output = self.encoder(embedding_output, attention_mask, train)
        pooled_output = self.pooler(encoder_output[:, 0])
        return encoder_output, nnx.tanh(pooled_output)
