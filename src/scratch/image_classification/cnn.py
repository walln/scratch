"""CNN model implementation.

This file contains the implementation of a simple CNN model.
The CNN model is designed for image classification tasks and is particularly suited for
simple datasets such as MNIST.

Reference:
    LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning
    applied to document recognition. Proceedings of the IEEE.
"""

from dataclasses import dataclass
from functools import partial

from flax import nnx

from scratch.datasets.image_classification_dataset import (
    mnist_dataset,
)
from scratch.image_classification.trainer import (
    ImageClassificationParallelTrainer,
    ImageClassificationParallelTrainerConfig,
)
from scratch.utils.logging import console


@dataclass
class CNNConfig:
    """Configuration for the CNN model."""

    num_classes: int = 10
    """Number of classes in the dataset."""
    input_shape: tuple[int, int, int] = (28, 28, 1)
    """Number of input channels."""


class CNN(nnx.Module):
    """A simple CNN model.

    This class implements a basic Convolutional Neural Network (CNN) for image
    classification. It consists of two convolutional layers followed by two
    fully connected layers.
    """

    def __init__(self, config: CNNConfig, *, rngs: nnx.Rngs):
        """Initializes the simple CNN model.

        Args:
            config: Configuration for the model.
            rngs: Random number generators.
        """
        self.conv1 = nnx.Conv(config.input_shape[-1], 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
        self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
        self.linear2 = nnx.Linear(256, config.num_classes, rngs=rngs)

    def __call__(self, x):
        """Forward pass of the model.

        Args:
            x: Input array of shape (batch_size, height, width, channels).

        Returns:
            Output array of shape (batch_size, num_classes).
        """
        x = self.avg_pool(nnx.relu(self.conv1(x)))
        x = self.avg_pool(nnx.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = nnx.relu(self.linear1(x))
        x = self.linear2(x)
        return x


"""
Trains a simple CNN model on the MNIST dataset.

Running on my RTX 3080ti, and loading the dataset from a memmapped file, the training
only takes ~20 seconds for an epoch the full training split and reaches ~98% accuracy on
the test split.

Will likely be even faster on a TPU or a multi-GPU setup. The trainer naturally supports
SPMD parallelism and can be easily adapted to use multiple devices with no changes.

The simple CNN model is too simple for decent MFU and the bottleneck is the data loading
"""
if __name__ == "__main__":
    console.log("Loading dataset")
    batch_size = 64
    dataset = mnist_dataset(
        batch_size=batch_size,
        shuffle=True,
    )

    console.log(f"Dataset metadata: {dataset.metadata}")
    assert dataset.test is not None, "Test dataset is None"

    console.log("Configuring model")
    model_config = CNNConfig(
        num_classes=dataset.metadata.num_classes, input_shape=(28, 28, 1)
    )
    model = CNN(model_config, rngs=nnx.Rngs(0))

    trainer_config = ImageClassificationParallelTrainerConfig(
        batch_size=batch_size, epochs=1
    )

    # To enable WANDB logging, uncomment the following lines
    # logger = WeightsAndBiasesLogger(
    #     "walln_scratch_cnn_mnist", model_config, trainer_config
    # )
    # And comment out the following line
    logger = None

    trainer = ImageClassificationParallelTrainer[CNN](
        model, trainer_config, logger=logger
    )
    trainer.train_and_evaluate(dataset.train, dataset.test)
