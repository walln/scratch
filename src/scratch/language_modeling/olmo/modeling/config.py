"""Configuration for OLMo."""

from dataclasses import dataclass


@dataclass
class OLMoConfig:
    """Configuration for OLMo.

    Attributes:
        d_model: the hidden size of the model
        n_heads: the number of attention heads
        n_layers: the number of transformer blocks
        mlp_ratio: the ratio of the hidden size of the MLP to the hidden size of
          the model
        max_sequence_length: the maximum sequence length
        vocab_size: the number of tokens in the vocabulary
        embedding_size: the number of embeddings (token space)
        block_group_size: the number of blocks to group together, this is for
          distributed training purposes.
        residual_dropout: the dropout rate for the residual connections in
          the transformer blocks (MLP and attention layers)
        embedding_dropout: the dropout rate for the embedding layer
        attention_dropout: the dropout rate for the attention layers
        rope: whether to use rotary position embeddings
        weight_tying: whether to tie output linear layer weights to the input embedding
        include_bias: whether to include bias in the linear layers
          (bias is near 0 for large models - see palm paper)
        use_multi_query_attention: whether to use multi-query attention
        clip_qkv: the clip coefficient for the QKV projections
        attention_layer_norm_with_affine: whether to use affine transformation for QK
          norms
        weight_tying: whether to tie output linear layer weights to the input embedding
        scale_logits: whether to scale the logits in the attention layer
          by 1 / sqrt(d_model).
        init_device: the device to use for initialization
    """

    # Model shape
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    mlp_ratio: int = 8
    max_sequence_length: int = 2048
    vocab_size: int = 50304
    embedding_size: int = 50304  # nearest multiple of 128 to vocab_size this
    # can really improve throughput according to the paper
    block_group_size: int = 1

    # Dropout configuration
    residual_dropout: float = 0.0
    embedding_dropout: float = 0.0
    attention_dropout: float = 0.0

    # Model features
    use_rope: bool = True  # alibi also used but mutually exclusive with rope
    weight_tying: bool = True
    include_bias: bool = False
    use_multi_query_attention: bool = False
    clip_qkv: float | None = None
    attention_layer_norm_with_affine: bool = True
    scale_logits: bool = False

    @property
    def effective_n_kv_heads(self) -> int:
        """Return the effective number of key-value heads.

        This is the number of heads used in the key and value projections. This
        will change from the number of heads if using multi-query attention.
        """
        if self.use_multi_query_attention:
            return 1
        else:
            return self.n_heads
