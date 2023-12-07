from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

from jax.numpy import ndarray
from jax_dataloader import DataLoader as JaxDataLoader

from datasets import load_dataset

T = TypeVar("T")


class Dataloader(Generic[T]):
    def __init__(self, loader, transform):
        self.dataloader = loader
        self.transform = transform
        self.iterator = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self) -> T:
        try:
            batch = next(self.iterator)
            transformed_batch = self.transform(batch)
            return transformed_batch
        except StopIteration:
            self.iterator = iter(self.dataloader)
            raise StopIteration

    def __len__(self):
        return len(self.dataloader)


class ImageClassificationBatch:
    inputs: ndarray
    targets: ndarray


def mnist_dataset(batch_size=32, shuffle=True):
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


@dataclass
class Dataset(Generic[T]):
    """
    Data module class that contains loaders
    """

    batch_size: int
    train: Dataloader[T]
    test: Optional[Dataloader[T]]
    validation: Optional[Dataloader[T]]
