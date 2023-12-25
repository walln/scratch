"""BERT model."""
import dataclasses
import functools
from typing import Optional

import equinox as eqx
import jax
from jaxtyping import Array


@dataclasses.dataclass(frozen=True)
class BertBlockConfig:
    """BERT block configuration."""

    num_heads: int
    embedding_size: int
    dropout: float


class BertBlock(eqx.Module):
    """Bert block."""

    attention: eqx.nn.MultiheadAttention
    layer_norm_1: eqx.nn.LayerNorm
    forward: eqx.nn.MLP
    layer_norm_2: eqx.nn.LayerNorm

    def __init__(self, block_config: BertBlockConfig, *, key: Array):
        """Initialize BertBlock."""
        attention_key, forward_key = jax.random.split(key, num=2)
        hidden_size = int(4 * block_config.embedding_size)

        self.attention = eqx.nn.MultiheadAttention(
            num_heads=block_config.num_heads,
            query_size=block_config.embedding_size,
            dropout_p=block_config.dropout,
            key=attention_key,
            use_key_bias=False,
            use_value_bias=False,
            use_query_bias=False,
            use_output_bias=False,
        )

        self.layer_norm_1 = eqx.nn.LayerNorm(shape=block_config.embedding_size)
        self.forward = eqx.nn.MLP(
            in_size=block_config.embedding_size,
            out_size=block_config.embedding_size,
            width_size=hidden_size,
            depth=1,
            activation=functools.partial(jax.nn.gelu, approximate=True),
            use_bias=False,
            use_final_bias=False,
            key=forward_key,
        )
        self.layer_norm_2 = eqx.nn.LayerNorm(shape=block_config.embedding_size)

    def __call__(self, x, *, dropout: bool = False, key: Optional[Array] = None):
        """Forward pass through BertBlock."""
        attention_key, dropout_key = (
            (None, None) if key is None else jax.random.split(key, num=2)
        )
        # Self-attention
        x_norm = jax.vmap(self.layer_norm_1)(x)
        x = x + self.attention(
            query=x_norm, key_=x_norm, value=x_norm, key=attention_key
        )

        # MLP
        x_norm = jax.vmap(self.layer_norm_2)(x)
        x = x + jax.vmap(self.forward)(x_norm, key=dropout_key)
        return x


@dataclasses.dataclass(frozen=True)
class BertConfig:
    """BERT configuration."""

    vocab_size: int
    embedding_size: int
    num_blocks: int
    num_heads: int
    dropout: float
    max_length: int


class Bert(eqx.Module):
    """BERT model."""

    token_embed: eqx.nn.Embedding
    position_embed: eqx.nn.Embedding
    blocks: eqx.nn.Sequential
    head: eqx.nn.Linear
    layer_norm: eqx.nn.LayerNorm

    def __init__(self, model_config: BertConfig, *, key: Array):
        """Initialize Bert model."""
        token_embed_key, position_embed_key, head_key, *block_key = jax.random.split(
            key, num=model_config.num_blocks + 3
        )

        self.token_embed = eqx.nn.Embedding(
            num_embeddings=model_config.vocab_size,
            embedding_size=model_config.embedding_size,
            key=token_embed_key,
        )
        self.position_embed = eqx.nn.Embedding(
            num_embeddings=model_config.max_length,
            embedding_size=model_config.embedding_size,
            key=position_embed_key,
        )

        block_config = BertBlockConfig(
            num_heads=model_config.num_heads,
            dropout=model_config.dropout,
            embedding_size=model_config.embedding_size,
        )
        self.blocks = eqx.nn.Sequential(
            [BertBlock(block_config=block_config, key=key) for key in block_key]
        )
