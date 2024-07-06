"""Simple self-attention layer computes attention scores and outputs.

Computes self-attention mechanism by generating query, key, and value tensors
from the input and calculating attention scores. The results are used to produce
the final output.

Based on the original self-attention mechanism proposed in [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
"""

import jax
import jax.numpy as jnp
from flax import nnx


class SelfAttention(nnx.Module):
    """Simple self-attention layer computes attention scores and outputs."""

    def __init__(self, input_dim: int, *, rngs: nnx.Rngs):
        """Initialize self-attention layer.

        Args:
            input_dim: The input dimension.
            rngs: Random number generators for initializing parameters.
        """
        self.input_dim = input_dim
        self.key = nnx.Linear(input_dim, input_dim, rngs=rngs)
        self.query = nnx.Linear(input_dim, input_dim, rngs=rngs)
        self.value = nnx.Linear(input_dim, input_dim, rngs=rngs)
        self.softmax = jax.nn.softmax

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
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
        scores = jnp.matmul(queries, keys.transpose((0, 2, 1))) / jnp.sqrt(
            self.input_dim
        )
        attention = self.softmax(scores, axis=-1)

        # Compute output
        output = jnp.matmul(attention, values)
        return output, attention
