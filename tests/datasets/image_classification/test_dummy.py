"""Test the dummy dataset."""

from torch import Tensor

from scratch.datasets.image_classification_dataset import (
    dummy_image_classification_dataset,
)


def test_loading():
    """Test that the dummy dataset can be loaded."""
    batch_size = 4
    shape = (32, 32, 3)
    num_samples = 100
    dataset = dummy_image_classification_dataset(
        batch_size=batch_size, shape=shape, num_samples=num_samples
    )

    assert dataset.metadata.num_classes == 10
    assert dataset.batch_size == batch_size

    # Check that the dataset can be loaded
    first = next(iter(dataset.train))
    assert first["image"].numpy().shape == (
        batch_size,
        *shape,
    )
    assert isinstance(first["image"], Tensor)

    assert first["label"].numpy().shape == (batch_size, dataset.metadata.num_classes)
    assert isinstance(first["label"], Tensor)
