"""Test the dummy dataset."""

from scratch.datasets.image_classification_dataset import (
    dummy_image_classification_dataset,
)


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
    assert first["image"].shape == (
        batch_size,
        *shape,
    )
    assert first["label"].shape == (batch_size, dataset.metadata.num_classes)
