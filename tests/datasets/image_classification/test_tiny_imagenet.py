"""Test the tiny_imagenet dataset."""

from scratch.datasets.image_classification_dataset import (
    patch_datasets_warning,
    tiny_imagenet_dataset,
)


def test_loading():
    """Test that the tiny_imagenet dataset can be loaded."""
    patch_datasets_warning()  # This is a workaround for a bug in the datasets library
    batch_size = 4
    dataset = tiny_imagenet_dataset(batch_size=batch_size)

    assert dataset.metadata.num_classes == 200
    assert dataset.batch_size == batch_size

    # Check that the dataset can be loaded
    first = next(iter(dataset.train))
    assert first["image"].numpy().shape == (
        batch_size,
        64,
        64,
        3,
    ), f"Shape: {first['image'].numpy().shape}"
    assert first["label"].numpy().shape == (batch_size, dataset.metadata.num_classes)
