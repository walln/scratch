"""Dataset utilities for sequence classification.

This module provides utilities for loading and preparing sequence classification
datasets. It includes functions for loading datasets from Hugging Face datasets
library, creating dummy datasets, and transforming datasets for training.

Example:
    To load the IMDb dataset:

    >>> imdb = imdb_dataset(batch_size=32)

    To create a dummy sequence classification dataset:

    >>> dummy = dummy_sequence_classification_dataset(batch_size=32)

    To patch the warning message for datasets caused by a bug with recent PyTorch:

    >>> patch_datasets_warning()

"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import torch
from datasets import IterableDataset, load_dataset
from torch.utils.data import DataLoader as TorchDataLoader
from transformers import BertTokenizer

from scratch.datasets.dataset import DataLoader, Dataset


class SequenceClassificationBatch(TypedDict):
    """Batch of sequences and labels.

    The batch contains an array of sequences (e.g., token IDs) shaped
    (batch_size, sequence_length) and an array of labels shaped (batch_size,).
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    label: torch.Tensor


@dataclass
class SequenceClassificationDatasetMetadata:
    """Metadata for sequence classification datasets."""

    num_classes: int
    input_shape: tuple[int]
    name: str
    vocab_size: int


def create_dataset(
    metadata: SequenceClassificationDatasetMetadata,
    train_data,
    test_data,
    batch_size: int,
    transform: Callable[[SequenceClassificationBatch], SequenceClassificationBatch]
    | None,
    collate_fn: Callable | None = None,
):
    """Create a Dataset object for sequence classification.

    Args:
        metadata: the metadata for the dataset
        train_data: the training data
        test_data: the test data
        transform: the transformation function to apply to the data
        batch_size: the batch size
        collate_fn: the collate function for the data loader

    Returns:
        The dataset
    """
    train_loader = TorchDataLoader(
        train_data,  # type: ignore - PyTorch types are incompatible
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    test_loader = TorchDataLoader(
        test_data,  # type: ignore - PyTorch types are incompatible
        batch_size=batch_size,
        collate_fn=collate_fn,
    )

    train_loader = DataLoader[SequenceClassificationBatch](
        loader=train_loader, transform=transform
    )
    test_loader = DataLoader[SequenceClassificationBatch](
        loader=test_loader, transform=transform
    )

    return Dataset[SequenceClassificationBatch, SequenceClassificationDatasetMetadata](
        batch_size=batch_size,
        train=train_loader,
        test=test_loader,
        validation=None,
        metadata=metadata,
    )


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

    if prepare:
        data = data.map(prepare).with_format("torch")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    data = data.map(tokenize_function, batched=True)

    return data.with_format("torch")


def dummy_sequence_classification_dataset(
    batch_size=32,
    shuffle=True,
    num_samples=128,
    num_classes=2,
    sequence_length=128,
    vocab_size=100,
):
    """Create a dummy sequence classification dataset.

    Args:
        batch_size: the batch size
        shuffle: whether to shuffle the dataset
        num_samples: the number of samples in the dataset
        num_classes: the number of classes
        sequence_length: the length of the sequences
        vocab_size: the size of the vocabulary
    """
    metadata = SequenceClassificationDatasetMetadata(
        num_classes=num_classes,
        input_shape=(sequence_length,),
        name="dummy",
        vocab_size=vocab_size,
    )

    def gen():
        for _ in range(num_samples):
            input_ids = np.random.randint(0, vocab_size, size=(sequence_length,))
            attention_mask = np.ones(sequence_length, dtype=np.int64)
            label = np.random.randint(0, num_classes)
            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
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
        input_ids, attention_mask, label, token_type_ids = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["label"],
            batch["token_type_ids"],
        )
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        attention_mask = torch.tensor(attention_mask, dtype=torch.int64)
        label = torch.tensor(label, dtype=torch.int64)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.int64)
        return SequenceClassificationBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            label=label,
            token_type_ids=token_type_ids,
        )

    return create_dataset(
        metadata=metadata,
        train_data=data,
        test_data=data,
        transform=transform,
        batch_size=batch_size,
    )


def imdb_dataset(batch_size=32, shuffle=True, tokenizer=None):
    """Load the IMDb dataset and return a Dataset object.

    Args:
        batch_size: the batch size
        shuffle: whether to shuffle the dataset
        tokenizer: the tokenizer to use for the dataset. If None, the default
                   BERT tokenizer will be used.
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def prepare(sample):
        input_ids, attention_mask, labels = (
            sample["input_ids"],
            sample["attention_mask"],
            sample["label"],
        )
        token_type_ids = torch.zeros_like(input_ids, dtype=torch.int64)
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        attention_mask = torch.tensor(attention_mask, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)
        (
            sample["input_ids"],
            sample["attention_mask"],
            sample["label"],
            sample["token_type_ids"],
        ) = (
            input_ids,
            attention_mask,
            labels,
            token_type_ids,
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
        num_classes=2, input_shape=(128,), name="imdb", vocab_size=len(tokenizer)
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
