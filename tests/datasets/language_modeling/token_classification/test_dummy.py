"""Test the dummy dataset."""

from scratch.datasets.token_classification_dataset import (
    dummy_token_classification_dataset,
)


def test_loading():
    """Test that the dummy dataset can be loaded."""
    batch_size = 4
    num_samples = 100
    dataset = dummy_token_classification_dataset(
        batch_size=batch_size, num_samples=num_samples
    )

    assert dataset.batch_size == batch_size

    # Check that the dataset can be loaded
    first = next(iter(dataset.train))

    input_ids = first["input_ids"]
    attention_mask = first["attention_mask"]
    labels = first["labels"]

    assert input_ids.numpy().shape == (batch_size, dataset.metadata.sequence_length)
    assert attention_mask.numpy().shape == (
        batch_size,
        dataset.metadata.sequence_length,
    )
    assert labels.numpy().shape == (batch_size, dataset.metadata.sequence_length)
