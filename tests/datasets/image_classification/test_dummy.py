"""Test the dummy dataset."""

from scratch.datasets.image_classification_dataset import (
    dummy_image_classification_dataset,
)
from torch import Tensor


def test_loading():
    """Test that the dummy dataset can be loaded."""
    batch_size = 4
    shape = (32, 32, 3)
    dataset = dummy_image_classification_dataset(batch_size=batch_size, shape=shape)

    assert dataset.metadata.num_classes == 10
    assert dataset.batch_size == batch_size
    assert len(dataset.train) == 100 // batch_size

    # Check that the dataset can be loaded
    first = next(iter(dataset.train))
    assert first["image"].numpy().shape == (
        batch_size,
        *shape,
    )
    assert isinstance(first["image"], Tensor)

    assert first["label"].numpy().shape == (batch_size, dataset.metadata.num_classes)
    assert isinstance(first["label"], Tensor)
