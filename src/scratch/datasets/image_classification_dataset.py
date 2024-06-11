"""Dataset utilities for image classification."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import DataLoader as TorchDataLoader

from scratch.datasets.dataset import (
    DataLoader,
    Dataset,
)


class ImageClassificationBatch(TypedDict):
    """Batch of images and labels.

    The batch contains an array of images shaped (batch_size, height, width, channels)
    and an array of one hot encoded labels shaped (batch_size, num_classes).
    """

    image: Tensor
    label: Tensor


@dataclass
class ImageClassificationDatasetMetadata:
    """Metadata for image classification datasets."""

    num_classes: int
    input_shape: tuple[int, int, int]
    name: str


def create_dataset(
    metadata: ImageClassificationDatasetMetadata,
    train_data,
    test_data,
    transform: Callable[[ImageClassificationBatch], ImageClassificationBatch],
    batch_size: int,
    collate_fn: Callable | None = None,
    *,
    shuffle: bool,
):
    """Create a Dataset object for image classification.

    Args:
        metadata: the metadata for the dataset
        train_data: the training data
        test_data: the test data
        transform: the transformation function to apply to the data
        batch_size: the batch size
        collate_fn: the collate function for the data loader
        shuffle: whether to shuffle the data

    Returns:
        The dataset
    """
    train_loader = TorchDataLoader(
        train_data,  # type: ignore - PyTorch types are incompatible
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    test_loader = TorchDataLoader(
        test_data,  # type: ignore - PyTorch types are incompatible
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )

    train_loader = DataLoader[ImageClassificationBatch](
        loader=train_loader, transform=transform
    )
    test_loader = DataLoader[ImageClassificationBatch](
        loader=test_loader, transform=transform
    )

    return Dataset[ImageClassificationBatch, ImageClassificationDatasetMetadata](
        batch_size=batch_size,
        train=train_loader,
        test=test_loader,
        validation=None,
        metadata=metadata,
    )


def dummy_image_classification_dataset(
    batch_size=32, shuffle=True, num_samples=100, num_classes=10, shape=(28, 28, 1)
):
    """Create a dummy image classification dataset.

    Args:
    ----
        batch_size: the batch size
        shuffle: whether to shuffle the dataset
        num_samples: the number of samples in the dataset
        num_classes: the number of classes
        shape: the shape of the images
    """
    metadata = ImageClassificationDatasetMetadata(
        num_classes=10, input_shape=(28, 28, 1), name="dummy"
    )

    # Just create an array of random numbers as the dummy dataset
    data = np.random.rand(num_samples, *shape)
    labels = np.random.randint(0, num_classes, num_samples)

    # Now turn into a list of dicts with each dict being one sample and label
    data = [
        {"image": image, "label": label}
        for image, label in zip(data, labels, strict=False)
    ]

    def transform(batch: ImageClassificationBatch):
        """A image classification batch transformation function.

        Image classification batch transformation functions must take a batch of data
        and return a batch of ImageClassificationBatch objects. Where the batch of data
        has an image that is a numpy array of shape
        (batch_size, height, width, channels) in float32 format and a label that is
        a one hot encoded numpy array of shape (batch_size, num_classes) in
        int32 format.
        """
        image, label = batch["image"], batch["label"]
        image = image.numpy().astype(np.float32)
        label = (
            F.one_hot(label, num_classes=metadata.num_classes).numpy().astype(np.int32)
        )
        return ImageClassificationBatch(image=image, label=label)

    return create_dataset(
        metadata=metadata,
        train_data=data,
        test_data=data,
        transform=transform,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def mnist_dataset(batch_size=32, shuffle=True):
    """Load the MNIST dataset and return a Dataset object.

    Args:
    ----
        batch_size: the batch size
        shuffle: whether to shuffle the dataset
    """

    def prepare(sample):
        images, labels = sample["image"], sample["label"]
        # Ensure the images are float tensors
        images = images.to(torch.float32)
        # Normalize the images
        images = images / 255.0
        # Convert labels to one-hot encoding
        labels = labels.to(torch.int64)  # Ensure labels are int32 tensors
        labels = F.one_hot(labels, num_classes=10).to(torch.int32)

        sample["image"], sample["label"] = images, labels
        return sample

    train_data = (
        load_dataset("mnist", split="train", trust_remote_code=True)
        .with_format("torch")
        .map(prepare)
    )
    test_data = (
        load_dataset("mnist", split="test", trust_remote_code=True)
        .with_format("torch")
        .map(prepare)
    )

    metadata = ImageClassificationDatasetMetadata(
        num_classes=10, input_shape=(28, 28, 1), name="mnist"
    )

    def transform(batch: ImageClassificationBatch):
        """MNIST is greyscale so we need to add a channel dimension."""
        image, label = batch["image"], batch["label"]
        image = image.unsqueeze(-1)
        return ImageClassificationBatch(image=image, label=label)

    return create_dataset(
        metadata=metadata,
        train_data=train_data,
        test_data=test_data,
        transform=transform,
        batch_size=batch_size,
        shuffle=shuffle,
    )
