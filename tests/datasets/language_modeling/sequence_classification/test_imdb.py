"""Test the imdb dataset."""

from scratch.datasets.sequence_classification_dataset import (
    imdb_dataset,
)


def test_loading():
    """Test that the imdb dataset can be loaded."""
    batch_size = 4
    dataset = imdb_dataset(batch_size=batch_size)

    assert dataset.batch_size == batch_size

    # Check that the dataset can be loaded
    first = next(iter(dataset.train))

    input_ids = first["input_ids"]
    labels = first["label"]

    assert input_ids.numpy().shape == (batch_size, dataset.metadata.max_sequence_length)
    assert labels.numpy().shape == (batch_size, dataset.metadata.num_classes)
