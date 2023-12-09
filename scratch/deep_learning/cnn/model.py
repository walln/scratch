from flax import linen as nn
from flax.linen.initializers import kaiming_normal


class CNN(nn.Module):
    """
    A simple CNN model for image classification.
    """

    num_classes: int

    @nn.compact
    def __call__(self, x, train=True, **kwargs):
        x = nn.Conv(features=32, kernel_size=(3, 3), kernel_init=kaiming_normal())(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3), kernel_init=kaiming_normal())(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((-1, x.shape[-3] * x.shape[-2] * x.shape[-1]))
        x = nn.Dense(features=256, kernel_init=kaiming_normal())(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_classes)(x)

        return x
