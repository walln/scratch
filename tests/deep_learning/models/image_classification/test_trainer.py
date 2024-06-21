"""Test the trainer functions as expected."""

import jax.numpy as jnp
from flax import nnx
from scratch.datasets.image_classification_dataset import (
    dummy_image_classification_dataset,
)
from scratch.image_classification.trainer import (
    ImageClassificationParallelTrainer,
    ImageClassificationParallelTrainerConfig,
)


class DummyModel(nnx.Module):
    """Dummy model for testing."""

    def __init__(
        self, num_classes: int, input_shape: tuple[int, int, int], *, rngs: nnx.Rngs
    ):
        """Initialize a dummy model with a single conv layer and a linear layer.

        Args:
            num_classes (int): The number of classes.
            input_shape (tuple[int, int, int]): The input shape of the model.
            rngs (nnx.Rngs): The random number generators
        """
        self.num_classes = num_classes
        self.conv = nnx.Conv(
            in_features=input_shape[-1], out_features=1, kernel_size=(1, 1), rngs=rngs
        )
        self.linear = nnx.Linear(
            input_shape[0] * input_shape[1], num_classes, rngs=rngs
        )

    def __call__(self, x: jnp.ndarray):
        """Forward pass of the model.

        Args:
            x (jnp.ndarray): The input tensor.

        Returns:
            The output tensor.
        """
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        x = nnx.relu(x)
        return x


def test_basic_usage():
    """Test the basic usage of the trainer. This is a smoke test."""
    batch_size = 2
    shape = (28, 28, 1)
    dataset = dummy_image_classification_dataset(
        shape=shape, batch_size=batch_size, num_samples=16
    )
    dummy_model = DummyModel(
        num_classes=dataset.metadata.num_classes,
        input_shape=shape,
        rngs=nnx.Rngs(0),
    )
    config = ImageClassificationParallelTrainerConfig(batch_size=batch_size, epochs=1)
    trainer = ImageClassificationParallelTrainer(
        model=dummy_model, trainer_config=config
    )
    assert trainer.model.num_classes == dataset.metadata.num_classes
    trainer.train_and_evaluate(train_loader=dataset.train, test_loader=dataset.test)
