"""Tests for the ViT model."""

from flax import nnx
from scratch.datasets.image_classification_dataset import (
    dummy_image_classification_dataset,
)
from scratch.image_classification.vision_transformer import (
    VisionTransformer,
    VisionTransformerConfig,
)


def test_forward_pass():
    """Test the forward pass of the model."""
    batch_size = 2
    shape = (224, 224, 3)
    dataset = dummy_image_classification_dataset(
        shape=shape, batch_size=batch_size, num_samples=16
    )
    model_config = VisionTransformerConfig(
        num_classes=dataset.metadata.num_classes,
        input_shape=shape,
    )
    dummy_model = VisionTransformer(model_config=model_config, rngs=nnx.Rngs(0))
    batch = next(iter(dataset.train))
    y = dummy_model(batch["image"].numpy())
    assert y.shape == (batch_size, dataset.metadata.num_classes)
