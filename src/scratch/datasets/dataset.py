"""Custom datasets for use with the scratch framework."""

import dataclasses
from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar

from torch.utils.data import DataLoader as TorchDataLoader

T = TypeVar("T")
B = TypeVar("B")
M = TypeVar("M")

"""
A Note on dataloading:
----------------------
I tried a huge amount to optimize dataloading and experimented with the various options.
According to my naive benchmarks on single-node training, If you are using PyTorch,
just use the data loader as is. Writing a prefetcher and custom loader is not worth it
as it is really hard to improve the performance with a custom collator and fetcher.

When using jax it is still probably worth it to just use the default collator and
fetcher and just cast the tensors to jax arrays. The performance difference is
negligible and writing a custom collator and fetcher that is faster than the
default one is really hard.

I tried github.com/google/grain but is it not quite ready as the docs are non-existent
and it demands the length of the dataset to be known which is not always the case for
iterable datasets. I also tried using the torch DataLoader with the jax arrays by
loading them from disk in the jax format but you then have to deal with the collator
which brings back the same problem as before.

Tensorflow datasets are an option and I did try them because the performance is good.
But TFDS is just kind of gross.

My custom prefetchers are not faster than the default PyTorch DataLoader. So the final
conclusion is that the default PyTorch DataLoader is the best option for pretty much any
use case that does not require the maximum performance possible and guaranteed
reproducibility.
"""

"""
A Note on transformations:
--------------------------
The huggingface datasets library is really good for loading datasets and transforming
them. However, it has a few poorly documented issues. Primarily about how batched
transformations are stored in disk/memory. It is really important that if you do not
save the transformed dataset to disk as its own dataset and rely on the cache that you
ensure that the transformations are done exclusively in torch.Tensor format.
This is because they can be copied with zero cost.

Additionally, it is a good idea to consider storage layout when transforming the data.
For instance, training image models typically peform better with NHWC layout. This is
because the convolutional kernels are optimized for this layout. However, theis format
is not contiguous in memory so it can be slower for the arrow tables to read. So it is
a good idea to keep the data in the NCHW format in the arrow tables and then transform
it to NHWC when loading it into the model. This is because the NHWC format is faster for
the model to read and the NCHW format is faster for the arrow tables to read.
"""


class DataLoader(Generic[B]):
    """Custom DataLoader that applies a transformation to each batch."""

    def __init__(
        self, loader: TorchDataLoader, transform: Callable[[Any], B] | None = None
    ):
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
            transformed_batch = self.transform(batch) if self.transform else batch
            yield transformed_batch

    def __len__(self):
        """Return the number of batches."""
        return len(self.loader)


@dataclasses.dataclass
class Dataset(Generic[T, M]):
    """Data module class that contains loaders."""

    batch_size: int
    metadata: M
    train: DataLoader[T]
    test: DataLoader[T] | None
    validation: DataLoader[T] | None
