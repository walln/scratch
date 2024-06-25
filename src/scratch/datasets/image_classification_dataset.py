"""Dataset utilities for image classification."""

import inspect
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from datasets import IterableDataset, load_dataset
from torch import Tensor
from torch.utils.data import DataLoader as TorchDataLoader

from scratch.datasets.dataset import (
    DataLoader,
    Dataset,
)


def nchw_to_nhwc(image: Tensor) -> Tensor:
    """Convert a batch of images from NCHW to NHWC format."""
    return image.permute(0, 2, 3, 1)


def patch_datasets_warning():
    """Patch the warning message for datasets.

    The warning message is due to a bug? in the huggingface datasets library.
    It might be logically correct but pytorch warns on using the spread operator to copy
    tensors rather than using the clone method.
    """
    warning_message = "To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor)."  # noqa: E501
    warnings.filterwarnings("ignore", message=warning_message, category=UserWarning)

    def filter_specific_warning(warning):
        """Custom filter function to ignore a warning coming from a specific file."""
        if warning.category is UserWarning:
            # Get the current frame
            frame = inspect.currentframe()
            # Go back to the frame where the warning was issued
            while frame:
                filename = frame.f_code.co_filename
                if filename.endswith("datasets/formatting/torch_formatter.py"):
                    print("HERE")

                    return True
                frame = frame.f_back
        return False

    # Register the custom filter
    warnings.filterwarnings("ignore", category=UserWarning, module=r".*")
    warnings.showwarning = (
        lambda message, category, filename, lineno, file=None, line=None: None
        if filter_specific_warning(
            warnings.WarningMessage(message, category, filename, lineno)
        )
        else warnings.showwarning(message, category, filename, lineno)
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
    batch_size: int,
    transform: Callable[[ImageClassificationBatch], ImageClassificationBatch] | None,
    collate_fn: Callable | None = None,
):
    """Create a Dataset object for image classification.

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


def load_hf_dataset(
    dataset_name: str,
    dataset_split: str,
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
        prepare: the prepare function to apply to the dataset
        validate: the validate function to apply to the dataset
        shuffle: whether to shuffle the dataset

    Returns:
        The IterableDataset object
    """
    data = load_dataset(
        dataset_name, split=dataset_split, trust_remote_code=True
    ).with_format("torch")

    if shuffle:
        data = data.shuffle().with_format("torch")

    if validate:
        data = data.filter(validate)

    if prepare:
        data = data.map(prepare)

    if shuffle:
        data = data.shuffle().with_format("torch")

    return data


def dummy_image_classification_dataset(
    batch_size=32, shuffle=True, num_samples=128, num_classes=10, shape=(28, 28, 1)
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

    def gen():
        for _ in range(num_samples):
            image = np.random.rand(*shape)
            label = np.random.randint(0, num_classes)
            yield {"image": image, "label": label}

    data = IterableDataset.from_generator(gen)

    if shuffle:
        data = data.shuffle(buffer_size=num_samples)

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
        image = image.to(torch.float32)
        label = F.one_hot(label, num_classes=metadata.num_classes).to(torch.int32)
        return ImageClassificationBatch(image=image, label=label)

    return create_dataset(
        metadata=metadata,
        train_data=data,
        test_data=data,
        transform=transform,
        batch_size=batch_size,
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

    train_data, test_data = (
        load_hf_dataset("mnist", "train", prepare=prepare, shuffle=shuffle),
        load_hf_dataset("mnist", "test", prepare=prepare, shuffle=shuffle),
    )

    metadata = ImageClassificationDatasetMetadata(
        num_classes=10, input_shape=(28, 28, 1), name="mnist"
    )

    def transform(batch: ImageClassificationBatch):
        """MNIST is greyscale so we need to add a channel dimension."""
        image, label = batch["image"], batch["label"]
        image = nchw_to_nhwc(image)
        return ImageClassificationBatch(image=image, label=label)

    return create_dataset(
        metadata=metadata,
        train_data=train_data,
        test_data=test_data,
        transform=transform,
        batch_size=batch_size,
    )


# TODO: Albumentations augmentations
def tiny_imagenet_dataset(batch_size=32, shuffle=True):
    """Load the Tiny ImageNet dataset and return a Dataset object.

    Args:
    ----
        batch_size: the batch size
        shuffle: whether to shuffle the dataset
    """

    def prepare(sample):
        images, labels = sample["image"], sample["label"]
        # Ensure the images are float tensors
        images = images.clone().detach().to(torch.float32)
        # Normalize the images
        images = images / 255.0
        # Convert labels to one-hot encoding
        labels = labels.clone().detach().to(torch.int64)  # Ensure labels are int32
        labels = F.one_hot(labels, num_classes=200).to(torch.int32)

        sample["image"], sample["label"] = images, labels
        return sample

    def validate(sample):
        return (
            sample["image"].shape == (3, 64, 64)
            and torch.isnan(sample["image"]).sum() == 0
            and torch.isinf(sample["image"]).sum() == 0
        )

    train_data, test_data = (
        load_hf_dataset(
            "zh-plus/tiny-imagenet",
            "train",
            prepare=prepare,
            validate=validate,
            shuffle=shuffle,
        ),
        load_hf_dataset(
            "zh-plus/tiny-imagenet",
            "valid",
            prepare=prepare,
            validate=validate,
            shuffle=shuffle,
        ),
    )

    metadata = ImageClassificationDatasetMetadata(
        num_classes=200, input_shape=(64, 64, 3), name="tiny_imagenet"
    )

    def transform(batch: ImageClassificationBatch):
        image, label = batch["image"], batch["label"]
        image = nchw_to_nhwc(image)
        return ImageClassificationBatch(image=image, label=label)

    return create_dataset(
        metadata=metadata,
        train_data=train_data,
        test_data=test_data,
        batch_size=batch_size,
        transform=transform,
    )
