"""Dataset utilities for token classification.

This module provides utilities for loading and preparing token classification
datasets. It includes functions for loading datasets from Hugging Face datasets
library, creating dummy datasets, and transforming datasets for training.

Example:
    To create a dummy sequence classification dataset:

    >>> dummy = dummy_token_classification_dataset(batch_size=32)
"""

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import torch
from datasets import IterableDataset

from scratch.datasets.dataset import create_dataset


@dataclass
class TokenClassificationMetadata:
    """Metadata for token classification datasets."""

    num_labels: int
    sequence_length: int
    vocab_size: int
    name: str


class TokenClassificationBatch(TypedDict):
    """Batch of input_ids, attention_mask, and labels for token classification."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def dummy_token_classification_dataset(
    batch_size=32,
    num_samples=128,
    sequence_length=64,
    vocab_size=30522,
    num_labels=5,
    shuffle=True,
):
    """Create a dummy token classification dataset.

    Args:
    ----
        batch_size: the batch size
        num_samples: the number of samples in the dataset
        sequence_length: the length of the sequences
        vocab_size: the size of the vocabulary
        num_labels: the number of labels
        shuffle: whether to shuffle the dataset
    """
    metadata = TokenClassificationMetadata(
        num_labels=num_labels,
        sequence_length=sequence_length,
        name="dummy",
        vocab_size=vocab_size,
    )

    def gen():
        for _ in range(num_samples):
            input_ids = np.random.randint(0, vocab_size, size=(sequence_length,))
            attention_mask = np.ones(sequence_length)
            labels = np.random.randint(0, num_labels, size=(sequence_length,))
            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

    data = IterableDataset.from_generator(gen)

    if shuffle:
        data = data.shuffle(buffer_size=num_samples)

    def transform(batch):
        input_ids, attention_mask, labels = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return TokenClassificationBatch(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    return create_dataset(
        metadata=metadata,
        train_data=data,
        test_data=data,
        transform=transform,
        batch_size=batch_size,
    )
