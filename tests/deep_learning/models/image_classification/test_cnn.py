"""Tests for the CNN model."""

from flax import nnx
from scratch.datasets.image_classification_dataset import (
    dummy_image_classification_dataset,
)
from scratch.image_classification.cnn import CNN, CNNConfig
from scratch.image_classification.trainer import (
    ImageClassificationParallelTrainer,
    ImageClassificationParallelTrainerConfig,
)


def test_forward_pass():
    """Test the forward pass of the cnn."""
    batch_size = 2
    shape = (28, 28, 1)
    dataset = dummy_image_classification_dataset(
        shape=shape, batch_size=batch_size, num_samples=16
    )
    model_config = CNNConfig(
        input_shape=shape,
        num_classes=dataset.metadata.num_classes,
    )
    dummy_model = CNN(config=model_config, rngs=nnx.Rngs(0))
    batch = next(iter(dataset.train))
    y = dummy_model(batch["image"])
    assert y.shape == (batch_size, dataset.metadata.num_classes)


def test_basic_training():
    """Test the basic usage of the cnn."""
    batch_size = 2
    shape = (28, 28, 1)
    dataset = dummy_image_classification_dataset(
        shape=shape, batch_size=batch_size, num_samples=16
    )
    model_config = CNNConfig(
        input_shape=shape,
        num_classes=dataset.metadata.num_classes,
    )
    dummy_model = CNN(config=model_config, rngs=nnx.Rngs(0))
    config = ImageClassificationParallelTrainerConfig(batch_size=batch_size, epochs=1)
    trainer = ImageClassificationParallelTrainer(
        model=dummy_model, trainer_config=config
    )
    trainer.train_and_evaluate(train_loader=dataset.train, test_loader=dataset.test)
