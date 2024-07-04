"""Test the wikitext dataset."""

from scratch.datasets.causal_langauge_modeling_dataset import (
    wikitext2_dataset,
)


def test_loading():
    """Test that the wikitext2 dataset can be loaded."""
    batch_size = 4
    dataset = wikitext2_dataset(batch_size=batch_size, shuffle=False)

    assert dataset.batch_size == batch_size

    # Check that the dataset can be loaded
    first = next(iter(dataset.train))

    input_ids = first["input_ids"]
    attention_mask = first["attention_mask"]
    labels = first["labels"]

    assert input_ids.numpy().shape == (batch_size, dataset.metadata.max_sequence_length)
    assert attention_mask.numpy().shape == (
        batch_size,
        dataset.metadata.max_sequence_length,
        dataset.metadata.max_sequence_length,
    )
    assert labels.numpy().shape == (batch_size, dataset.metadata.max_sequence_length)
