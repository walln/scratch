"""Tests for the ResNet model."""

from flax import nnx
from scratch.datasets.image_classification_dataset import (
    dummy_image_classification_dataset,
)
from scratch.image_classification.resnet import ResNet, ResNetConfig


def test_forward_pass():
    """Test the forward pass of the ResNet."""
    batch_size = 2
    shape = (28, 28, 1)
    dataset = dummy_image_classification_dataset(
        shape=shape, batch_size=batch_size, num_samples=16
    )
    model_config = ResNetConfig.from_preset(
        preset_name="resnet18",
        num_classes=dataset.metadata.num_classes,
        input_channels=shape[-1],
    )
    dummy_model = ResNet(config=model_config, rngs=nnx.Rngs(0))
    batch = next(iter(dataset.train))
    y = dummy_model(batch["image"])
    assert y.shape == (batch_size, dataset.metadata.num_classes)
