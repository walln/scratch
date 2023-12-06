from flax import linen as nn


class CNNOld(nn.Module):
    """
    A simple CNN model for image classification.
    """

    @nn.compact
    def __call__(self, x, train=True, **kwargs):
        # Convolutional layer: inherently supports batching
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Another convolutional layer
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((-1, x.shape[-3] * x.shape[-2] * x.shape[-1]))

        # Dense layers: also inherently support batching
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)

        return x


class CNN(nn.Module):
    """
    A simple CNN model for image classification.
    """

    num_classes: int

    @nn.compact
    def __call__(self, x, train=True, **kwargs):
        # Convolutional layer: inherently supports batching
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Another convolutional layer
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((-1, x.shape[-3] * x.shape[-2] * x.shape[-1]))

        # Dense layers: also inherently support batching
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_classes)(x)

        return x
