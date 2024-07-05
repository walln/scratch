"""OLMo model implementation."""

from typing import NamedTuple

import jax.numpy as jnp
from flax import nnx

from scratch.language_modeling.olmo.modeling.blocks.sequential import SequentialBlock
from scratch.language_modeling.olmo.modeling.config import OLMoConfig
from scratch.language_modeling.olmo.utils.numerical_stability import ensure_finite

base_config = OLMoConfig()


class OLMoForwardResult(NamedTuple):
    """Forward pass result for OLMo model.

    Attributes:
        logits: A tensor of shape (batch_size, seq_len, vocab_size) representing the
            log probs for the next token before normalization through log softmax.
        hidden_states: Hidden states from each block.
    """

    logits: jnp.ndarray
    hidden_states: tuple[jnp.ndarray] | None


class OLMo(nnx.Module):
    """OLMo model."""

    def __init__(self, config: OLMoConfig, *, rngs: nnx.Rngs):
        """Initialize the model.

        Args:
            config: The model configuration.
            rngs: The random number generators.
        """
        self.config = config

        if (
            self.config.embedding_size is not None
            and self.config.embedding_size != self.config.vocab_size
        ):
            if self.config.embedding_size < self.config.vocab_size:
                raise ValueError(
                    "embedding_size must be greater than or equal to vocab_size"
                )
        elif self.config.embedding_size % 128 != 0:
            # for throughput reasons, we want the embedding size to be a multiple of 128
            raise ValueError("embedding_size must be a multiple of 128")

        self.wte = nnx.Embed(
            config.embedding_size or config.vocab_size,
            config.d_model,
            rngs=rngs,
        )
        self.emb_drop = nnx.Dropout(config.embedding_dropout, rngs=rngs)
        self.ln_f = nnx.LayerNorm(config.d_model, rngs=rngs)

        self.blocks = [
            SequentialBlock(i, config, rngs=rngs) for i in range(config.n_layers)
        ]

        self.ff_out = nnx.Linear(
            config.d_model,
            config.embedding_size or config.vocab_size,
            use_bias=config.include_bias,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: jnp.ndarray,
        input_embeddings: jnp.ndarray | None = None,
        attention_mask: jnp.ndarray | None = None,
        attention_bias: jnp.ndarray | None = None,
        past_key_values: list[jnp.ndarray] | None = None,
        *,
        last_logits_only: bool = False,
        output_hidden_states: bool = False,
    ) -> OLMoForwardResult:
        """Forward pass through the OLMo model.

        Args:
            input_ids: The input token IDs. (batch_size, seq_len)
            input_embeddings: The input embeddings. (batch_size, seq_len, d_model)
                if provided, this will be used instead of the embeddings from the
                token IDs.
            attention_mask: The attention mask. (batch_size, seq_len)
                Tensor indicating which input ids should be attended to.
            attention_bias: The attention bias. (batch_size, 1, seq_len, seq_len)
                Tensor indicating the introduction of causal bias.
            past_key_values: The past key values. Pre-computed key and values for each
                attention layer. (batch_size, n_layers, 2, n_heads, seq_len, head_dim)
            last_logits_only: Whether to return only the last logits, for fast decoding.
            output_hidden_states: Whether to output hidden states.

        Returns:
            The logits, attention key values, and hidden states.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )

        if past_key_values:
            assert len(past_key_values) == self.config.n_layers

        if input_embeddings is None:
            batch_size, seq_len = input_ids.shape
        else:
            batch_size, seq_len = input_embeddings.shape[:2]

        x = self.wte(input_ids) if input_embeddings is None else input_embeddings
        x = self.emb_drop(x)

        # Attention masking
        if attention_mask is not None:
            attention_mask = attention_mask.astype(dtype=jnp.float32).view(
                batch_size, -1
            )[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * jnp.finfo(
                attention_mask.dtype
            ).min

        # Merge mask
        if (
            attention_bias is not None
            or attention_mask is not None
            or past_key_values is not None
        ):
            if attention_bias and attention_bias.dtype in (jnp.int8, jnp.bool):
                attention_bias = attention_bias.astype(dtype=jnp.float32)

                def mask_attention_bias(attention_bias):
                    min_value = jnp.finfo(attention_bias.dtype).min
                    attention_bias = jnp.where(
                        attention_bias == 0.0, min_value, attention_bias
                    )
                    return attention_bias

                attention_bias = mask_attention_bias(attention_bias)

            # Fix shape and dtype
            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            elif past_key_values is not None:
                mask_len = past_key_values[0][0].shape[-2] + seq_len

            if attention_bias is not None:
                attention_bias = attention_bias[:, :, :mask_len, :mask_len].astype(
                    dtype=jnp.float32
                )

            # add in bias
            if attention_bias is not None:
                attention_bias = attention_bias + attention_mask
                ensure_finite(attention_bias, check_neg_inf=True, check_pos_inf=False)

        all_hidden_states = []

        for block_idx, block in enumerate(self.blocks):
            if output_hidden_states:
                # add hidden states
                all_hidden_states.append(x)

            layer_past = None if past_key_values is None else past_key_values[block_idx]

            # shape: (batch_size, seq_len, d_model)
            x = block(
                x,
                attention_bias=attention_bias,
                layer_past=layer_past,
            )

        if last_logits_only:
            x = x[:, -1, :]
            x = jnp.expand_dims(x, axis=1)

        x = self.ln_f(x)
        if output_hidden_states:
            all_hidden_states.append(x)

        logits = self.ff_out(x)

        if self.config.scale_logits:
            jnp.multiply(logits, 1 / jnp.sqrt(self.config.d_model))

        return OLMoForwardResult(
            logits=logits,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
        )
