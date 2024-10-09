"""Mamba2 model implementation.

Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured
State Space Duality

This module contains the implementation of the Mamba-2 model.
https://arxiv.org/abs/2405.21060

This file contains the implementation of Mamba-2, a state space model (SSM) that
incorporates the structured state space duality (SSD) framework. Mamba-2 is a
refinement of the original Mamba model, achieving 2-8X faster training speeds
while maintaining competitive performance with Transformers on language modeling tasks.

The implementation is based on the paper "Transformers are SSMs: Generalized Models
and Efficient Algorithms Through Structured State Space Duality"
by Tri Dao and Albert Gu.

Reference:
    Dao, T., & Gu, A. (2024).
    Transformers are SSMs: Generalized Models and Efficient Algorithms Through
    Structured State Space Duality.
    arXiv preprint arXiv:2405.21060. https://arxiv.org/abs/2405.21060

Mamba-2 introduces several improvements over Mamba-1. The primary enhancement is the
adoption of the SSD algorithm, which allows the model to leverage both the efficiency
of state space models and the computational benefits of matrix multiplications.
This results in significantly faster training times. The SSD layer in Mamba-2 restricts
the state space model's recurrence matrices to a scalar times identity structure,
reducing complexity and improving efficiency. Additionally, Mamba-2 supports multi-head
SSMs, similar to multi-head attention mechanisms in Transformers, allowing for larger
state dimensions and enhanced model expressivity.

The SSD algorithm works by providing two equivalent interpretations of the model:
one as a structured matrix defining the sequence transformation and another as a
chunkwise algorithm. This duality allows the model to maintain efficient FLOP counts
and utilize optimized matrix multiplications, achieving a balance between computational
efficiency and model performance.

To actually train the model, a training loop can be performed using a CLMTrainer.

This module also includes the necessary components for incorporating the SSD algorithm,
enabling efficient computation of SSD layers. The architectural changes in Mamba-2
include parallel production of SSM parameters with the input, simplifying scaling and
tensor parallelism. The sharding constraints needed to perform this are not currently
implemented.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from einops import rearrange, repeat
from flax import nnx


def pad_or_truncate_to_length(x: jnp.ndarray, target_length: int):
    """Pad or truncate the last dimension of a tensor to a target length.

    Args:
        x: input tensor of shape [batch, ..., length]
        target_length: target length of the last dimension

    Returns:
        padded or truncated tensor
    """
    current_length = x.shape[-1]
    if current_length < target_length:
        # Pad
        pad_width = target_length - current_length
        return jnp.pad(x, ((0, 0), (0, 0), (pad_width, 0)))
    elif current_length > target_length:
        # Truncate
        return x[:, :, -target_length:]
    else:
        # No change needed
        return x


@dataclass
class MambaConfig:
    """Mamba configuration.

    Args:
        d_model: model dimension (D)
        n_layers: number of mamba layers in the model
        d_state: state dimension (N)
        d_conv: convolution kernel size
        expand: expansion factor (E)
        head_dim: head dimension (P)
        chunk_size: matrix partition size (Q)
        vocab_size: vocabulary size
        pad_vocab_size_multiplier: padding
        d_inner: inner dimension
        n_heads: number of heads
    """

    d_model: int
    n_layers: int = 24
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    head_dim: int = 64
    chunk_size: int = 64
    vocab_size: int = 50277
    pad_vocab_size_multiplier: int = 16

    def __post_init__(self):
        """Compute inner dimension and number of heads."""
        self.d_inner = self.d_model * self.expand
        assert self.d_inner % self.head_dim == 0
        self.n_heads = self.d_inner // self.head_dim
        if self.vocab_size % self.pad_vocab_size_multiplier != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiplier
                - self.vocab_size % self.pad_vocab_size_multiplier
            )


class InferenceCache(NamedTuple):
    """Inference Cache.

    Attributes:
        conv_state (batch_size, d_inner + 2 * d_state, d_conv): convolution state
        ssm_state (batch_size, n_heads, head_dim, d_state): SSM state
    """

    conv_state: jnp.ndarray
    ssm_state: jnp.ndarray

    @staticmethod
    def allocate(batch_size: int, config: MambaConfig):
        """Allocate InferenceCache.

        Args:
            batch_size: batch size
            config: MambaConfig

        Returns:
            InferenceCache
        """
        return InferenceCache(
            jnp.zeros((batch_size, config.d_inner + 2 * config.d_state, config.d_conv)),
            jnp.zeros((batch_size, config.n_heads, config.head_dim, config.d_state)),
        )


class DepthwiseConv1D(nnx.Module):
    """Depthwise convolution 1D layer.

    Depthwise convolution is a type of convolution where the number of input channels
    is equal to the number of output channels. This layer applies a depthwise
    convolution to the input tensor, followed by a pointwise convolution with a
    kernel size of 1.

    Attributes:
        layer: depthwise convolution layer
    """

    def __init__(
        self, conv_dim: int, kernel_size: int, padding: int, *, rngs: nnx.Rngs
    ):
        """Initialize DepthwiseConv1D.

        Args:
            conv_dim: number of input channels
            kernel_size: kernel size of the depthwise convolution
            padding: padding of the depthwise convolution
            rngs: random number generators
        """
        self.layer = nnx.Conv(
            in_features=conv_dim,
            out_features=conv_dim,
            kernel_size=(kernel_size,),
            feature_group_count=conv_dim,
            padding=[(padding, padding)],
            # kaiming normal is relu optimal
            kernel_init=nnx.initializers.he_normal(),
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray):
        """Forward pass through the depthwise convolution layer.

        Args:
            x (batch_size, seq_len, d_model): input tensor

        Returns:
            output tensor of shape (batch_size, seq_len, d_model)
        """
        x = x.transpose(0, 2, 1)
        out = self.layer(x)
        out = out.transpose(0, 2, 1)
        return out


class MambaLayer(nnx.Module):
    """Mamba Layer.

    A Mamba layer consists of an input projection, a depthwise convolution, and an
    SSD function.

    Attributes:
        config: MambaConfig
        in_proj: input projection layer
        conv: depthwise convolution layer
        dt_bias: bias for the dt parameter
        A_log: log of the A parameter
        D: D parameter
        norm: normalization layer
        out_proj: output projection layer
    """

    def __init__(self, config: MambaConfig, *, rngs: nnx.Rngs):
        """Initialize MambaLayer.

        Args:
            config: MambaConfig
            rngs: random number generators
        """
        self.config = config

        d_in_proj = 2 * config.d_inner + 2 * config.d_state + config.n_heads
        self.in_proj = nnx.Linear(
            config.d_model,
            d_in_proj,
            use_bias=False,
            # glorot uniform balances variance between input and output dimensions
            kernel_init=nnx.initializers.glorot_uniform(),
            rngs=rngs,
        )

        conv_dim = config.d_inner + 2 * config.d_state
        self.conv = DepthwiseConv1D(
            conv_dim=conv_dim,
            kernel_size=config.d_conv,
            padding=config.d_conv - 1,
            rngs=rngs,
        )

        self.dt_bias = nnx.Param(
            # after softplus dt needs to remain positive while still being small
            nnx.initializers.constant(0.1)(rngs(), (config.n_heads,)),
            rngs=rngs,
        )
        self.A_log = nnx.Param(
            # initialize A_log to -1.0 so that after exp it is close to 0.0 but still
            # positive - this is since A = -exp(A_log)
            nnx.initializers.constant(-1.0)(rngs(), (config.n_heads,)),
            rngs=rngs,
        )
        self.D = nnx.Param(
            # small stable scaling factor for residual (not sure what would be better)
            nnx.initializers.constant(0.1)(rngs(), (config.n_heads,)),
            rngs=rngs,
        )
        self.norm = nnx.RMSNorm(config.d_inner, rngs=rngs)
        self.out_proj = nnx.Linear(
            config.d_inner,
            config.d_model,
            use_bias=False,
            # glorot uniform balances variance between input and output dimensions
            kernel_init=nnx.initializers.glorot_uniform(),
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray, cache: InferenceCache | None = None):
        """Forward pass through a Mamba layer.

        Args:
            x (batch_size, seq_len, d_model): input tensor. Seqlen should be a
                multiple of chunk_size.
            cache: hidden states for inference step. If None, hidden states are
                initialized to zeros.

        Returns:
            output tensor of shape (batch_size, seq_len, d_model) and updated hidden
            states.
        """
        if cache:
            return self.step(x, cache)

        A = -jnp.exp(self.A_log.value)
        zxbcdt = self.in_proj(x)

        z, xbc, dt = jnp.split(
            zxbcdt,
            [
                self.config.d_inner,
                zxbcdt.shape[-1] - self.config.n_heads,
            ],
            axis=-1,
        )

        dt = jax.nn.softplus(dt + self.dt_bias.value)

        # Pad or truncate the xbc tensor to match the conv kernel size
        xbc_rearranged = rearrange(xbc, "b l d -> b d l")
        conv_state = pad_or_truncate_to_length(xbc_rearranged, self.config.d_conv)

        # apply 1d convolution and silu activation
        xbc_conv = self.conv(xbc.transpose(0, 2, 1)).transpose(0, 2, 1)
        xbc_silu = jax.nn.silu(xbc_conv[:, : x.shape[1], :])

        # split the conv state into the conv kernel and the conv state
        sizes_xbc = jnp.array(
            [self.config.d_inner, self.config.d_state, self.config.d_state]
        )
        split_indices_xbc = jnp.cumsum(sizes_xbc)[:-1]
        x, b, c = jnp.split(xbc_silu, split_indices_xbc, axis=-1)

        # rearrange x
        x = rearrange(x, "b l (h p) -> b l h p", p=self.config.head_dim)

        # apply ssd function
        y, ssm_state = ssd(
            x * jnp.expand_dims(dt, axis=-1),
            A * dt,
            rearrange(b, "b l n -> b l 1 n"),
            rearrange(c, "b l n -> b l 1 n"),
            self.config.chunk_size,
        )

        # Combine the output of the ssd function with the input and rearrange
        y = y + x * jnp.expand_dims(self.D.value, axis=-1)
        y = rearrange(y, "b l h p -> b l (h p)")

        # apply the output projection
        y = self.norm(y, z)
        y = self.out_proj(y)

        hidden_state = InferenceCache(conv_state, ssm_state)
        return y, hidden_state

    def step(
        self, x: jnp.ndarray, cache: InferenceCache
    ) -> tuple[jnp.ndarray, InferenceCache]:
        """Forward pass through a single step of the Mamba layer.

        This function implements a single step of the Mamba layer, which consists
        of a projection, a depthwise convolution, and an SSD function. It takes in
        the input tensor, x, and the hidden states, cache, and returns the output
        tensor, y, and the updated hidden states. The hidden states are updated
        using the SSD function. This function is used when the hidden states are
        provided.

        Args:
            x (batch_size, 1, d_model): input tensor
            cache: hidden states for inference step. If None, hidden states are
                initialized to zeros.

        Returns:
            output tensor of shape (batch_size, 1, d_model) and updated hidden
            states.
        """
        assert x.shape[1] == 1, "Only supports single token inputs"

        # Squeeze dimension 1 from x
        x_squeezed = jnp.squeeze(x, axis=1)

        # Project input using in_proj
        zxbcdt = self.in_proj(x_squeezed)  # (batch, d_in_proj)

        # Split zxbcdt into z, xBC, and dt
        sizes = [
            self.config.d_inner,
            self.config.d_inner + 2 * self.config.d_state,
            self.config.n_heads,
        ]
        indices = jnp.cumsum(jnp.array(sizes))[:-1]
        z, xBC, dt = jnp.split(zxbcdt, indices, axis=-1)

        conv_state = cache.conv_state
        ssm_state = cache.ssm_state

        # Advance convolution input
        conv_state = jnp.roll(conv_state, shift=-1, axis=-1)
        conv_state = conv_state.at[:, :, -1].set(xBC)

        # Convolution step
        conv_weight_rearranged = rearrange(self.conv.layer.kernel, "d 1 w -> d w")
        xBC = jnp.sum(conv_state * conv_weight_rearranged, axis=-1)
        xBC += self.conv.layer.bias
        xBC = jax.nn.silu(xBC)

        # Split xBC into x, B, and C
        sizes_xBC = [self.config.d_inner, self.config.d_state, self.config.d_state]
        indices_xBC = jnp.cumsum(jnp.array(sizes_xBC))[:-1]
        x, B, C = jnp.split(xBC, indices_xBC, axis=-1)

        # Exponential of A_log
        A = -jnp.exp(self.A_log.value)  # (nheads,)

        # SSM step
        dt = jax.nn.softplus(dt + self.dt_bias)  # (batch, nheads)
        dA = jnp.exp(dt * A)  # (batch, nheads)

        # Rearrange x
        x = rearrange(x, "b (h p) -> b h p", p=self.config.head_dim)

        # Compute dBx
        dBx = jnp.einsum("bh, bn, bhp -> bhpn", dt, B, x)

        # Update ssm_state
        ssm_state = ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx

        # Compute y
        y = jnp.einsum("bhpn, bn -> bhp", ssm_state, C)
        y = y + rearrange(self.D, "h -> h 1") * x

        # Rearrange y
        y = rearrange(y, "b h p -> b (h p)")

        # Apply normalization and output projection
        y = self.norm(y, z)
        y = self.out_proj(y)

        return y, InferenceCache(conv_state, ssm_state)


class Mamba2(nnx.Module):
    """Mamba2 model.

    Mamba2 is a variant of Mamba that uses a ssd function to compute the SSM
    states. This allows for more efficient computation of the SSM states, which
    can be beneficial for large models.

    Attributes:
        wte: embedding layer
        layers: list of MambaLayers
        norm: normalization layer
        head_norm: normalization layer for the head
        head: head layer
    """

    def __init__(self, config: MambaConfig, *, rngs: nnx.Rngs):
        """Initialize Mamba2.

        Args:
            config: MambaConfig
            rngs: random number generators
        """
        self.wte = nnx.Embed(config.vocab_size, config.d_model, rngs=rngs)
        self.layers = [MambaLayer(config, rngs=rngs) for _ in range(config.n_layers)]
        self.norm = nnx.RMSNorm(config.d_model, rngs=rngs, epsilon=1e-5)
        self.head_norm = nnx.RMSNorm(config.d_model, rngs=rngs, epsilon=1e-5)

    def __call__(
        self,
        input_ids: jnp.ndarray,
        hidden_states: list[InferenceCache] | list[None] | None = None,
    ):
        """Forward pass through the Mamba2 model.

        Args:
            input_ids (batch_size, seq_len): input ids
            hidden_states: list of hidden states for each layer. If None, hidden
                states are initialized to zeros.

        Returns:
            logits (batch_size, seq_len, vocab_size): logits
            hidden_states: list of updated hidden states for each layer
        """
        seq_len = input_ids.shape[1]

        if hidden_states is None:
            hidden_states = [None for _ in range(len(self.layers))]

        x = self.wte(input_ids)
        for i, layer in enumerate(self.layers):
            assert hidden_states is not None
            y, h = layer(self.norm(x), hidden_states[i])
            assert isinstance(h, InferenceCache) or h is None
            hidden_states[i] = h  # type: ignore
            x = x + y

        x = self.head_norm(x)
        # logits = self.head(x)
        logits = jnp.dot(x, self.wte.embedding.value.T)
        # instead of using a linear layer, we can just use the embedding matrix
        # this weight tying saves a lot of parameters. Could use a linear with tied
        # weights but the nnx docs are not clear on how to do this yet.
        return logits[:, :seq_len], hidden_states


def segsum(x: jnp.ndarray):
    """Stable segment sum calculation.

    Produces a 1-semiseperable matrix which is equivalent to a scalar SSM.

    Args:
        x (batch_size, seq_len, n_heads): input tensor

    Returns:
        output tensor of shape (batch_size, seq_len, n_heads)
    """
    T = x.shape[-1]
    x = repeat(x, "... d -> ... d e", e=T)
    mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool), -1)
    x = jnp.where(mask, x, 0)
    x_segsum = jnp.cumsum(x, axis=-2)
    mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool), 0)
    x_segsum = jnp.where(mask, x_segsum, -jnp.inf)
    return x_segsum


def ssd(
    x: jnp.ndarray,
    A: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    chunk_size: int,
    initial_states: jnp.ndarray | None = None,
):
    """Structured State Space Duality (SSD).

    This function implements the SSD algorithm for computing the SSM states. It
    takes in the input tensor, A, B, and C, and returns the output tensor, y, and
    the updated SSM states. The SSD algorithm is a generalization of the SSM
    algorithm to the case where the SSM states are not scalars. It is a
    structured matrix multiplication that is equivalent to a scalar SSM.

    Args:
        x: (batch, seq_len, n_heads, d_head)
        A: (batch, seq_len, n_heads)
        B: (batch, seq_len, n_heads, d_state)
        C: (batch, seq_len, n_heads, d_state)
        chunk_size: matrix partition size
        initial_states: (batch, 1, n_heads, d_state)

    Returns:
        y: (batch, seq_len, n_heads, d_head)
        state: (batch, seq_len, n_heads, d_state)
    """
    assert x.shape[1] % chunk_size == 0

    # Rearrange into chunks
    x, A, B, C = (
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    )

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = jnp.cumsum(A, axis=-1)

    # Compute intra-chunk state (diagonal blocks)
    L = jnp.exp(segsum(A))
    Y_diag = jnp.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # Compute intra-chunk state - the right term of low rank factorization of the
    # off diagonal blocks; B terms
    decay_states = jnp.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = jnp.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # Compute the inter-chunk SSM recurrence. Producing the correct SSM states at chunk
    # boundaries. This is the middle term of off diagonal blocks; A terms.
    if initial_states is None:
        initial_states = jnp.zeros_like(states[:, :1])

    states = jnp.concat([initial_states, states], axis=1)
    decay_chunk = jnp.exp(
        segsum(jnp.pad(A_cumsum[:, :, :, -1], ((0, 0), (0, 0), (1, 0))))
    )
    new_states = jnp.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # Compute state and output conversion per chunk
    # the left term of low rank factorization of the off diagonal blocks; C terms
    state_decay_out = jnp.exp(A_cumsum)
    Y_off = jnp.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

    # Add the output of intra-chunk and inter-chunk states
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    return Y, final_state


if __name__ == "__main__":
    config = MambaConfig(512)
    model = Mamba2(config, rngs=nnx.Rngs(jax.random.PRNGKey(0)))
    input_ids = jnp.zeros((1, 128), dtype=jnp.int32)
    logits, _ = model(input_ids)
    print(logits.shape)
    assert logits.shape == (1, 128, config.vocab_size)
