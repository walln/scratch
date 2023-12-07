from typing import Any

import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers as init


def kaiming_normal_init(rng: Any, shape: tuple, dtype: Any = jnp.float32):
    fan_in, fan_out = shape[-2:]
    return nn.initializers.kaiming_normal()


class CNN(nn.Module):
    """
    A simple CNN model for image classification.
    """

    num_classes: int

    @nn.compact
    def __call__(self, x, train=True, **kwargs):
        # Convolutional layer: inherently supports batching
        x = nn.Conv(features=32, kernel_size=(3, 3), kernel_init=init.kaiming_normal())(
            x
        )
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Another convolutional layer
        x = nn.Conv(features=64, kernel_size=(3, 3), kernel_init=init.kaiming_normal())(
            x
        )
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((-1, x.shape[-3] * x.shape[-2] * x.shape[-1]))

        # Dense layers: also inherently support batching
        x = nn.Dense(features=256, kernel_init=init.kaiming_normal())(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_classes)(x)

        return x
