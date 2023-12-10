from dataclasses import dataclass  # noqa: D100, I001
from typing import Generic, Optional, TypeVar

import jax
from datasets import load_dataset
from jax_dataloader import DataLoader as JaxDataLoader

T = TypeVar("T")


class Dataloader(Generic[T]):
    """Wrapper around DataLoader that applies a transform to the batch."""

    def __init__(self, loader, transform):
        """Initialize the dataloader.

        Args:
        ----
            loader: the dataloader to wrap
        transform: the transform to apply to the batch
        """
        self.dataloader = loader
        self.transform = transform
        self.iterator = iter(self.dataloader)

    def __iter__(self):
        """Return the iterator over the dataloader."""
        return self

    def __next__(self) -> T:
        """Return the next batch."""
        try:
            batch = next(self.iterator)
            transformed_batch = self.transform(batch)
            return transformed_batch
        except StopIteration:
            self.iterator = iter(self.dataloader)
            raise StopIteration

    def __len__(self):
        """Return the number of batches in the dataloader."""
        return len(self.dataloader)


class ImageClassificationBatch:
    """Batch of images and labels."""

    inputs: jax.numpy.ndarray
    targets: jax.numpy.ndarray


def mnist_dataset(batch_size=32, shuffle=True):
    """Load the MNIST dataset and return a Dataset object.

    Args:
    ----
        batch_size: the batch size
        shuffle: whether to shuffle the dataset
    """
    dataset = load_dataset("mnist")
    train_loader = JaxDataLoader(
        dataset["train"], "jax", batch_size=batch_size, shuffle=shuffle
    )
    test_loader = JaxDataLoader(
        dataset["test"], "jax", batch_size=batch_size, shuffle=shuffle
    )

    def transform_colnames(batch):
        # change shape from (batch_size, 28, 28) to (batch_size, 28, 28, 1)
        batch["image"] = batch["image"].reshape(
            batch["image"].shape[0], batch["image"].shape[1], batch["image"].shape[2], 1
        )

        # Scale pixel values from range [0, 255] to [0, 1]
        batch["image"] = batch["image"].astype(float) / 255.0
        return (batch["image"], batch["label"])

    train_loader = Dataloader(train_loader, transform_colnames)
    test_loader = Dataloader(test_loader, transform_colnames)

    return Dataset[ImageClassificationBatch](
        batch_size=batch_size, train=train_loader, test=test_loader, validation=None
    )


def cifar10_dataset(batch_size=32, shuffle=True):
    """Load the CIFAR10 dataset and return a Dataset object.

    Args:
    ----
        batch_size: the batch size
        shuffle: whether to shuffle the dataset
    """
    dataset = load_dataset("cifar10")
    train_loader = JaxDataLoader(
        dataset["train"], "jax", batch_size=batch_size, shuffle=shuffle
    )
    test_loader = JaxDataLoader(
        dataset["test"], "jax", batch_size=batch_size, shuffle=shuffle
    )

    def transform_colnames(batch):
        # Scale pixel values from range [0, 255] to [0, 1]
        batch["image"] = batch["image"].astype(float) / 255.0
        return (batch["image"], batch["label"])

    train_loader = Dataloader(train_loader, transform_colnames)
    test_loader = Dataloader(test_loader, transform_colnames)

    return Dataset[ImageClassificationBatch](
        batch_size=batch_size, train=train_loader, test=test_loader, validation=None
    )


@dataclass
class Dataset(Generic[T]):
    """Data module class that contains loaders."""

    batch_size: int
    train: Dataloader[T]
    test: Optional[Dataloader[T]]
    validation: Optional[Dataloader[T]]
