"""Dataset utilities for sequence classification.

This module provides utilities for loading and preparing sequence classification
datasets. It includes functions for loading datasets from Hugging Face datasets
library, creating dummy datasets, and transforming datasets for training.

Example:
    To load the IMDb dataset:

    >>> imdb = imdb_dataset(batch_size=32)

    To create a dummy sequence classification dataset:

    >>> dummy = dummy_sequence_classification_dataset(batch_size=32)
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from datasets import IterableDataset, load_dataset

from scratch.datasets.dataset import create_dataset
from scratch.datasets.utils import TokenizerMetadata, load_tokenizer


class SequenceClassificationBatch(TypedDict):
    """Batch of sequences and labels.

    The batch contains an array of sequences (e.g., token IDs) shaped
    (batch_size, sequence_length) and an array of labels shaped (batch_size,).
    """

    input_ids: torch.Tensor
    label: torch.Tensor


@dataclass
class SequenceClassificationDatasetMetadata:
    """Metadata for sequence classification datasets."""

    num_classes: int
    max_sequence_length: int
    name: str
    vocab_size: int
    tokenizer_metadata: TokenizerMetadata


def load_hf_dataset(
    dataset_name: str,
    dataset_split: str,
    tokenizer: Callable,
    *,
    prepare: Callable | None = None,
    validate: Callable | None = None,
    shuffle=True,
):
    """Load a dataset from the Hugging Face datasets library.

    Creates an IterableDataset object from the Hugging Face datasets library by
    streaming the data on the fly. New elements are fetched from the remote server
    as needed. Elements will go through the order of:
    - Loading the dataset
    - Shuffling the dataset
    - Validating the dataset
    - Preparing the dataset
    - Tokenizing the sequences

    Args:
        dataset_name: the name of the dataset
        dataset_split: the split of the dataset
        tokenizer: the tokenizer function to tokenize the sequences
        prepare: the prepare function to apply to the dataset
        validate: the validate function to apply to the dataset
        shuffle: whether to shuffle the dataset

    Returns:
        The IterableDataset object
    """
    data = load_dataset(
        dataset_name, split=dataset_split, trust_remote_code=True, streaming=True
    )

    if shuffle:
        data = data.shuffle().with_format("torch")

    if validate:
        data = data.filter(validate).with_format("torch")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    data = data.map(tokenize_function, batched=True).with_format("torch")

    if prepare:
        data = data.map(prepare).with_format("torch")

    return data.with_format("torch")


def dummy_sequence_classification_dataset(
    batch_size=32,
    shuffle=True,
    num_samples=128,
    num_classes=2,
    sequence_length=128,
    vocab_size=100,
    tokenizer_name: str = "bert-base-uncased",
):
    """Create a dummy sequence classification dataset.

    Args:
        batch_size: the batch size
        shuffle: whether to shuffle the dataset
        num_samples: the number of samples in the dataset
        num_classes: the number of classes
        sequence_length: the length of the sequences
        vocab_size: the size of the vocabulary
        tokenizer_name: the name of the tokenizer to use
    """
    tokenizer = load_tokenizer(tokenizer_name)
    metadata = SequenceClassificationDatasetMetadata(
        num_classes=num_classes,
        max_sequence_length=sequence_length,
        name="dummy",
        vocab_size=vocab_size,
        tokenizer_metadata=TokenizerMetadata.from_tokenizer(tokenizer, sequence_length),
    )

    def gen():
        for _ in range(num_samples):
            input_ids = np.random.randint(0, vocab_size, size=(sequence_length,))
            label = np.random.randint(0, num_classes)
            yield {
                "input_ids": input_ids,
                "label": label,
            }

    data = IterableDataset.from_generator(gen)

    if shuffle:
        data = data.shuffle(buffer_size=num_samples)

    def transform(batch: SequenceClassificationBatch):
        """A sequence classification batch transformation function.

        Sequence classification batch transformation functions must take a batch of data
        and return a batch of SequenceClassificationBatch objects. Where the batch of
        data has input_ids and attention_mask that are numpy arrays of shape
        (batch_size, sequence_length) in int64 format and a label that is
        a numpy array of shape (batch_size,) in int64 format.
        """
        input_ids, label = (
            batch["input_ids"],
            batch["label"],
        )
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        label = torch.tensor(label, dtype=torch.int64)
        label = F.one_hot(label, num_classes=num_classes).to(torch.int32)
        return SequenceClassificationBatch(
            input_ids=input_ids,
            label=label,
        )

    return create_dataset(
        metadata=metadata,
        train_data=data,
        test_data=data,
        transform=transform,
        batch_size=batch_size,
    )


def imdb_dataset(
    batch_size=32,
    shuffle=True,
    tokenizer_name: str = "bert-base-uncased",
    max_length=128,
):
    """Load the IMDb dataset and return a Dataset object.

    Args:
        batch_size: the batch size
        shuffle: whether to shuffle the dataset
        tokenizer_name: the tokenizer to use for the dataset. If None, the default
                        BERT tokenizer will be used.
        max_length: the maximum length of the sequences
    """
    tokenizer = load_tokenizer(tokenizer_name, max_length=max_length)

    def prepare(sample):
        input_ids, labels = (
            sample["input_ids"],
            sample["label"],
        )
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)
        labels = F.one_hot(labels, num_classes=2).to(torch.int32)

        (
            sample["input_ids"],
            sample["label"],
        ) = (
            input_ids,
            labels,
        )
        return sample

    train_data, test_data = (
        load_hf_dataset(
            "imdb", "train", tokenizer=tokenizer, prepare=prepare, shuffle=shuffle
        ),
        load_hf_dataset(
            "imdb", "test", tokenizer=tokenizer, prepare=prepare, shuffle=shuffle
        ),
    )

    metadata = SequenceClassificationDatasetMetadata(
        num_classes=2,
        max_sequence_length=max_length,
        name="imdb",
        vocab_size=len(tokenizer),
        tokenizer_metadata=TokenizerMetadata.from_tokenizer(tokenizer, max_length),
    )

    def transform(batch: SequenceClassificationBatch):
        """No transformation needed for sequence classification."""
        return batch

    return create_dataset(
        metadata=metadata,
        train_data=train_data,
        test_data=test_data,
        transform=transform,
        batch_size=batch_size,
    )
