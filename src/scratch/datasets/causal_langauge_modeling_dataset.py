"""Dataset utilities for causal language modeling.

This module provides utilities for loading and preparing causal language modeling
datasets. It includes functions for loading datasets from Hugging Face datasets
library, creating dummy datasets, and transforming datasets for training.

Example:
    To load the WikiText-2 dataset:

    >>> wikitext2 = wikitext2_dataset(batch_size=32)

    To create a dummy causal language modeling dataset:

    >>> dummy = dummy_clm_dataset(batch_size=32)
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import torch
from datasets import IterableDataset, load_dataset

from scratch.datasets.dataset import create_dataset
from scratch.datasets.utils import TokenizerMetadata, load_tokenizer


class CausalLanguageModelingBatch(TypedDict):
    """Batch of input sequences and labels for causal language modeling.

    The batch contains an array of input sequences (e.g., token IDs) shaped
    (batch_size, sequence_length) and an array of labels
    shaped (batch_size, sequence_length).
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


@dataclass
class CausalLanguageModelingDatasetMetadata:
    """Metadata for causal language modeling datasets."""

    vocab_size: int
    max_sequence_length: int
    name: str
    tokenizer_metadata: TokenizerMetadata


def load_hf_dataset(
    dataset_name: str,
    dataset_split: str,
    tokenizer: Callable,
    dataset_version: str | None = None,
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

    Args:
        dataset_name: the name of the dataset
        dataset_split: the split of the dataset
        tokenizer: the tokenizer function to tokenize the sequences
        dataset_version: the version of the dataset
        prepare: the prepare function to apply to the dataset
        validate: the validate function to apply to the dataset
        shuffle: whether to shuffle the dataset

    Returns:
        The IterableDataset object
    """
    data = load_dataset(
        dataset_name,
        split=dataset_split,
        trust_remote_code=True,
        streaming=True,
        name=dataset_version,
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


def dummy_clm_dataset(
    batch_size=32,
    shuffle=True,
    num_samples=128,
    vocab_size=100,
    sequence_length=128,
    tokenizer_name: str = "openai-community/gpt2",
):
    """Create a dummy causal language modeling dataset.

    Args:
        batch_size: the batch size
        shuffle: whether to shuffle the dataset
        num_samples: the number of samples in the dataset
        vocab_size: the size of the vocabulary
        sequence_length: the length of the sequences
        tokenizer_name: the tokenizer to tokenize the sequences. If None, the default
                        GPT2 tokenizer will be used.
    """
    tokenizer = load_tokenizer(tokenizer_name, max_length=sequence_length)
    metadata = CausalLanguageModelingDatasetMetadata(
        vocab_size=vocab_size,
        max_sequence_length=sequence_length,
        name="dummy",
        tokenizer_metadata=TokenizerMetadata.from_tokenizer(tokenizer, sequence_length),
    )

    def gen():
        for _ in range(num_samples):
            input_ids = np.random.randint(0, vocab_size, size=(sequence_length,))
            attention_mask = np.tril(np.ones((sequence_length, sequence_length)))
            labels = np.copy(input_ids)
            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

    data = IterableDataset.from_generator(gen)

    if shuffle:
        data = data.shuffle(buffer_size=num_samples)

    def transform(batch: CausalLanguageModelingBatch):
        """A causal language modeling batch transformation function.

        Causal language modeling batch transformation functions must take a batch of
        data and return a batch of CausalLanguageModelingBatch objects.
        Where the batch of data has input_ids, attention_mask, and labels that are
        numpy arrays of shape (batch_size, sequence_length) in int64 format.
        """
        input_ids, attention_mask, labels = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        attention_mask = torch.tensor(attention_mask, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)
        return CausalLanguageModelingBatch(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    return create_dataset(
        metadata=metadata,
        train_data=data,
        test_data=data,
        transform=transform,
        batch_size=batch_size,
    )


def wikitext2_dataset(
    batch_size=32,
    shuffle=True,
    tokenizer_name: str = "openai-community/gpt2",
    max_length=128,
):
    """Load the WikiText-2 dataset and return a Dataset object.

    Args:
        batch_size: the batch size.
        shuffle: whether to shuffle the dataset.
        tokenizer_name: the tokenizer to tokenize the sequences. If None, the default
                   GPT2 tokenizer will be used.
        max_length: the maximum length of the sequences.
    """
    tokenizer = load_tokenizer(tokenizer_name, max_length=max_length)

    def prepare(sample):
        input_ids = sample["input_ids"]
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        labels = input_ids.clone()
        # Make a lower triangular attention mask
        attention_mask = np.tril(np.ones((len(input_ids), len(input_ids))))
        attention_mask = torch.tensor(attention_mask, dtype=torch.int64)
        sample["input_ids"], sample["attention_mask"], sample["labels"] = (
            input_ids,
            attention_mask,
            labels,
        )
        return sample

    def validate(sample):
        return len(sample["text"])

    train_data, test_data = (
        load_hf_dataset(
            "Salesforce/wikitext",
            "train",
            tokenizer=tokenizer,
            prepare=prepare,
            validate=validate,
            shuffle=shuffle,
            dataset_version="wikitext-2-v1",
        ),
        load_hf_dataset(
            "Salesforce/wikitext",
            "test",
            tokenizer=tokenizer,
            prepare=prepare,
            validate=validate,
            shuffle=shuffle,
            dataset_version="wikitext-2-v1",
        ),
    )

    metadata = CausalLanguageModelingDatasetMetadata(
        vocab_size=tokenizer.vocab_size,
        max_sequence_length=max_length,
        name="wikitext2",
        tokenizer_metadata=TokenizerMetadata.from_tokenizer(tokenizer, max_length),
    )

    def transform(batch: CausalLanguageModelingBatch):
        """Create attention mask and labels for the batch."""
        return batch

    return create_dataset(
        metadata=metadata,
        train_data=train_data,
        test_data=test_data,
        transform=transform,
        batch_size=batch_size,
    )
