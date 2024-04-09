"""Custom datasets for use with the scratch framework."""

import dataclasses
from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar

import numpy as np
import torch.nn.functional as F
from datasets import load_dataset
from jaxtyping import Array
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

T = TypeVar("T")
B = TypeVar("B")


@dataclasses.dataclass
class ImageClassificationDatasetMetadata:
    """Metadata for image classification datasets."""

    num_classes: int
    input_shape: tuple[int, int, int]
    name: str


@dataclasses.dataclass
class BatchData(Generic[T]):
    """Wrapper around a batch of data."""

    data: T


@dataclasses.dataclass
class CustomImageClassificationBatch(BatchData[tuple[Array, Array]]):
    """Batch of images and labels."""

    def unpack(self):
        """Return a tuple of inputs and targets for unpacking.

        Returns:
        -------
        Tuple of inputs and targets.
        """
        return self.data


class CustomDataLoader(Generic[B]):
    """Custom DataLoader that applies a transformation to each batch."""

    def __init__(self, loader: DataLoader, transform: Callable[[Any], B]):
        """Create a CustomDataLoader.

        Args:
        ----
            loader (DataLoader): The original DataLoader.
            transform (Callable): The transformation function to apply to each batch.
        """
        self.loader = loader
        self.transform = transform

    def __iter__(self) -> Iterator[B]:
        """Iterate over the DataLoader."""
        for batch in self.loader:
            # Apply the transformation and yield a BatchData instance
            transformed_batch = self.transform(batch)
            yield transformed_batch

    def __len__(self):
        """Return the number of batches."""
        return len(self.loader)


# TODO: fix this to match others
def mnist_dataset(batch_size=32, shuffle=True):
    """Load the MNIST dataset and return a Dataset object.

    Args:
    ----
        batch_size: the batch size
        shuffle: whether to shuffle the dataset
    """
    train_data = load_dataset("mnist", split="train")
    test_data = load_dataset("mnist", split="test")

    metadata = ImageClassificationDatasetMetadata(
        num_classes=10, input_shape=(28, 28, 1), name="mnist"
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)  # type: ignore - PyTorch types are incompatible
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)  # type: ignore - PyTorch types are incompatible

    def transform_colnames(batch):
        # change shape from (batch_size, 28, 28) to (batch_size, 28, 28, 1)
        batch["image"] = (
            batch["image"]
            .reshape(
                batch["image"].shape[0],
                batch["image"].shape[1],
                batch["image"].shape[2],
                1,
            )
            .to_numpy()
        )

        # Scale pixel values from range [0, 255] to [0, 1]
        batch["image"] = batch["image"].astype(float) / 255.0
        batch["label"] = batch["label"].to_numpy()
        return CustomImageClassificationBatch(data=(batch["image"], batch["label"]))

    train_loader = CustomDataLoader[CustomImageClassificationBatch](
        loader=train_loader, transform=transform_colnames
    )
    test_loader = CustomDataLoader[CustomImageClassificationBatch](
        loader=test_loader, transform=transform_colnames
    )

    return CustomDataset[CustomImageClassificationBatch](
        batch_size=batch_size, train=train_loader, test=test_loader, validation=None
    ), metadata


def cifar10_dataset(batch_size=32, shuffle=True):
    """Load the CIFAR10 dataset and return a Dataset object.

    Args:
    ----
        batch_size: the batch size
        shuffle: whether to shuffle the dataset
    """
    dataset = load_dataset("cifar10").with_format("torch")

    metadata = ImageClassificationDatasetMetadata(
        num_classes=10, input_shape=(32, 32, 3), name="cifar10"
    )

    def custom_collate(batch):
        processed_batch = []
        for item in batch:
            image, label = item["img"], item["label"]

            # Ensure that the image is a 3-channel image
            if image.ndim == 2:
                # Add a channel dimension and convert to [H, W, C] by repeating
                # the channel
                image = image.unsqueeze(-1).repeat(1, 1, 3)

            # Convert to [H, W, C] if the image is in [C, H, W] format
            elif image.shape[0] == 3:
                image = image.permute(1, 2, 0)  # Reorder dimensions to [H, W, C]

            # If already in [H, W, C] but grayscale
            elif image.shape[-1] == 1:
                image = image.repeat(1, 1, 3)  # Repeat the channel 3 times

            processed_batch.append((image, label))

        # Use the default collate to stack the processed batch
        return default_collate(processed_batch)

    train_loader = DataLoader(
        dataset["train"],  # type: ignore - PyTorch types are incompatible
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate,
    )
    test_loader = DataLoader(
        dataset["test"],  # type: ignore - PyTorch types are incompatible
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate,
    )

    def transform_colnames(batch) -> CustomImageClassificationBatch:
        image, label = batch
        # Scale pixel values from range [0, 255] to [0, 1]
        image = image / 255.0
        label = F.one_hot(label, num_classes=metadata.num_classes)
        image = image.numpy().astype(np.float32)
        label = label.numpy().astype(np.float32)
        return CustomImageClassificationBatch(data=(image, label))

    train_loader = CustomDataLoader[CustomImageClassificationBatch](
        loader=train_loader, transform=transform_colnames
    )
    test_loader = CustomDataLoader[CustomImageClassificationBatch](
        loader=test_loader, transform=transform_colnames
    )

    return CustomDataset[CustomImageClassificationBatch](
        batch_size=batch_size, train=train_loader, test=test_loader, validation=None
    ), metadata


def tiny_imagenet_dataset(batch_size=16, shuffle=False):
    """Load the Tiny ImageNet dataset and return a Dataset object.

    Args:
    ----
        batch_size: the batch size
        shuffle: whether to shuffle the dataset
    """
    dataset = load_dataset("zh-plus/tiny-imagenet").with_format("torch")

    metadata = ImageClassificationDatasetMetadata(
        num_classes=200, input_shape=(64, 64, 3), name="tiny-imagenet"
    )

    def custom_collate(batch):
        processed_batch = []
        for item in batch:
            image, label = item["image"], item["label"]

            # Ensure that the image is a 3-channel image
            if image.ndim == 2:
                # Add a channel dimension and convert to [H, W, C] by repeating
                # the channel
                image = image.unsqueeze(-1).repeat(1, 1, 3)

            # Convert to [H, W, C] if the image is in [C, H, W] format
            elif image.shape[0] == 3:
                image = image.permute(1, 2, 0)  # Reorder dimensions to [H, W, C]

            # If already in [H, W, C] but grayscale
            elif image.shape[-1] == 1:
                image = image.repeat(1, 1, 3)  # Repeat the channel 3 times

            processed_batch.append((image, label))

        # Use the default collate to stack the processed batch
        return default_collate(processed_batch)

    train_loader = DataLoader(
        dataset["train"],  # type: ignore - PyTorch types are incompatible
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        collate_fn=custom_collate,
    )
    test_loader = DataLoader(
        dataset["valid"],  # type: ignore - PyTorch types are incompatible
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        collate_fn=custom_collate,
    )

    def transform_colnames(batch) -> CustomImageClassificationBatch:
        image, label = batch
        # Scale pixel values from range [0, 255] to [0, 1]
        image = image / 255.0
        label = F.one_hot(label, num_classes=metadata.num_classes)
        image = image.numpy().astype(np.float32)
        label = label.numpy().astype(np.float32)
        return CustomImageClassificationBatch(data=(image, label))

    train_loader = CustomDataLoader[CustomImageClassificationBatch](
        loader=train_loader, transform=transform_colnames
    )
    test_loader = CustomDataLoader[CustomImageClassificationBatch](
        loader=test_loader, transform=transform_colnames
    )
    return CustomDataset[CustomImageClassificationBatch](
        batch_size=batch_size, train=train_loader, test=test_loader, validation=None
    ), metadata


@dataclasses.dataclass
class CustomDataset(Generic[T]):
    """Data module class that contains loaders."""

    batch_size: int
    train: CustomDataLoader[T]
    test: CustomDataLoader[T] | None
    validation: CustomDataLoader[T] | None
