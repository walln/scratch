"""Test the MNIST dataset."""

from scratch.datasets.image_classification_dataset import mnist_dataset


def test_loading():
    """Test that the MNIST dataset can be loaded."""
    batch_size = 4
    dataset = mnist_dataset(batch_size=batch_size)

    assert dataset.metadata.num_classes == 10
    assert dataset.batch_size == batch_size

    # Check that the dataset can be loaded
    first = next(iter(dataset.train))
    assert first["image"].shape == (
        batch_size,
        28,
        28,
        1,
    )
    assert first["label"].shape == (batch_size, dataset.metadata.num_classes)
