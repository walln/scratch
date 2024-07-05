"""Activation functions."""

import jax
import jax.numpy as jnp
from flax import nnx


class SwiGLU(nnx.Module):
    """SwiGLU activation function.

    SwiGLU activation function is a Swish Gate Linear Unit activation function
    that is a combination of Swish and GLU activation functions.

    This linear gating function is smoother than ReLU and is non-monotonic.
    """

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply SwiGLU activation function.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        x, gate = jnp.split(x, 2, axis=-1)
        return jax.nn.silu(gate) * x
