"""Dataset utilities for question answering.

This module provides utilities for loading and preparing question answering
datasets. It includes functions for loading datasets from Hugging Face datasets
library, creating dummy datasets, and transforming datasets for training.

Example:
    To create a dummy question answering dataset:

    >>> dummy = dummy_question_answering_dataset(batch_size=32)
"""

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import torch
from datasets import IterableDataset

from scratch.datasets.dataset import create_dataset


@dataclass
class QuestionAnsweringMetadata:
    """Metadata for question answering datasets."""

    sequence_length: int
    vocab_size: int
    name: str


class QuestionAnsweringBatch(TypedDict):
    """Batch of input_ids, attention_mask, start_positions, and end_positions."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    start_positions: torch.Tensor
    end_positions: torch.Tensor


def dummy_question_answering_dataset(
    batch_size=32,
    num_samples=128,
    sequence_length=64,
    vocab_size=30522,
    shuffle=True,
):
    """Create a dummy question answering dataset.

    Args:
    ----
        batch_size: the batch size
        num_samples: the number of samples in the dataset
        sequence_length: the length of the sequences
        vocab_size: the size of the vocabulary
        shuffle: whether to shuffle the dataset
    """
    metadata = QuestionAnsweringMetadata(
        sequence_length=sequence_length,
        name="dummy",
        vocab_size=vocab_size,
    )

    def gen():
        for _ in range(num_samples):
            input_ids = np.random.randint(0, vocab_size, size=(sequence_length,))
            attention_mask = np.ones(sequence_length)
            # Ensure that start and end positions are within the sequence
            # length and start is before end
            start_positions = np.random.randint(0, sequence_length // 2)
            end_positions = np.random.randint(start_positions, sequence_length)
            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "start_positions": start_positions,
                "end_positions": end_positions,
            }

    data = IterableDataset.from_generator(gen)

    if shuffle:
        data = data.shuffle(buffer_size=num_samples)

    def transform(batch):
        input_ids, attention_mask, start_positions, end_positions = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["start_positions"],
            batch["end_positions"],
        )
        input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
        start_positions = torch.as_tensor(start_positions, dtype=torch.long)
        end_positions = torch.as_tensor(end_positions, dtype=torch.long)
        return QuestionAnsweringBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )

    return create_dataset(
        metadata=metadata,
        train_data=data,
        test_data=data,
        transform=transform,
        batch_size=batch_size,
    )
