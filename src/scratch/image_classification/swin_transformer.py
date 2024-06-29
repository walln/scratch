"""Swin Transformer Implementation.

This file contains the implementation of the Swin Transformer model.
The Swin Transformer is a hierarchical vision transformer that uses shifted
windows for computing self-attention, which improves both efficiency and scalability
for high-resolution vision tasks.

The implementation is based on the paper "Swin Transformer: Hierarchical Vision
Transformer using Shifted Windows"

Reference:
    Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021).
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.
    arXiv preprint arXiv:2103.14030. https://arxiv.org/abs/2103.14030
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import nnx
from scratch.datasets.image_classification_dataset import (
    dummy_image_classification_dataset,
)
from scratch.datasets.utils import patch_datasets_warning
from scratch.image_classification.trainer import (
    ImageClassificationParallelTrainer,
    ImageClassificationParallelTrainerConfig,
)
from scratch.utils.logging import console


def window_partition(x: jnp.ndarray, window_size: int) -> jnp.ndarray:
    """Partition the input image tensor into smaller non-overlapping windows.

    Args:
        x: The input image tensor with shape (B, H, W, C) where
           B is the batch size, H is the height, W is the width,
           and C is the number of channels.
        window_size: The size of each window (height and width).

    Returns:
        The partitioned windows with shape (num_windows, window_size, window_size, C)
        where num_windows = B * (H // window_size) * (W // window_size).
    """
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows


def window_reverse(
    windows: jnp.ndarray, window_size: int, H: int, W: int
) -> jnp.ndarray:
    """Reverse the partitioned windows back to the original image tensor.

    Args:
        windows: The partitioned windows with shape
                 (num_windows, window_size, window_size, C) where
                 num_windows = B * (H // window_size) * (W // window_size).
        window_size: The size of each window (height and width).
        H: The height of the original image.
        W: The width of the original image.

    Returns:
        The reconstructed image tensor with shape (B, H, W, C) where B is the batch
        size.
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.transpose(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    return x


@dataclass
class SwinTransformerConfig:
    """Configuration for the Swin Transformer model."""

    input_shape: tuple[int, int, int] = (224, 224, 3)
    """Input shape of the model."""
    num_classes: int = 10
    """Number of classes in the dataset."""
    patch_size: int = 4
    """Size of the patch."""
    embed_dim: int = 96
    """Dimension of the embedding."""
    depths: list[int] = field(default_factory=lambda: [2, 2, 6, 2])
    """Depths of the stages."""
    n_heads: list[int] = field(default_factory=lambda: [3, 6, 12, 24])
    """Number of heads in each stage."""
    window_size: int = 7
    """Size of the window."""
    mlp_ratio: float = 4.0
    """Ratio of the hidden layer to the embedding."""
    dropout: float = 0.1
    """Dropout rate."""
    attn_dropout: float = 0.1
    """Dropout rate for attention."""
    use_absolute_pos_embed: bool = False
    """Whether to use absolute positional embedding."""


class AdaptiveAvgPool1D(nnx.Module):
    """Adaptive Average Pooling Layer.

    This layer applies a 1D adaptive pooling operation to an input signal, which adjusts
    the input to a target output size. The pooling operation can be used to downsample
    or upsample the input while preserving the overall structure of the signal.
    """

    def __init__(self, output_size: int | tuple[int]):
        """Initializes the AdaptiveAvgPool1D layer.

        Args:
            output_size: Output size of the pooling operation
        """
        self.output_size = output_size

    def __call__(self, x: jnp.ndarray):
        """Forward pass of the layer.

        Args:
            x: Input array

        Returns:
            Output array
        """
        output_size = (
            (self.output_size,)
            if isinstance(self.output_size, int)
            else self.output_size
        )
        split = jnp.split(x, output_size[0], axis=1)
        stack = jnp.stack(split, axis=1)
        return stack.mean(axis=2)


class PatchMerging(nnx.Module):
    """Patch Merging Layer.

    This layer merges adjacent patches in an input feature map, reducing its spatial
    resolution while increasing the number of channels.
    """

    def __init__(self, input_resolution: tuple[int, int], dim: int, *, rngs: nnx.Rngs):
        """Initializes the PatchMerging module.

        Args:
          input_resolution: Input resolution
          dim: Dimension of the embedding
          rngs: Random number generators
        """
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nnx.Linear(4 * dim, 2 * dim, rngs=rngs)
        self.norm = nnx.LayerNorm(4 * dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        """Forward pass to merge patches.

        Args:
          x: Input array
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Input feature has invalid size"

        x = x.reshape(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = jnp.concatenate([x0, x1, x2, x3], axis=-1)
        x = x.reshape(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class WindowAttention(nnx.Module):
    """Window attention layer applying self-attention within local windows of the image.

    This module computes self-attention within non-overlapping windows of the input
    tensor, facilitating efficient attention computation.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        window_size: int,
        attn_dropout: float,
        dropout: float,
        *,
        rngs: nnx.Rngs,
    ):
        """Initializes the window attention layer.

        Args:
            embed_dim: Dimension of the embedding
            n_heads: Number of heads
            window_size: Size of the window
            attn_dropout: Dropout rate for attention
            dropout: Dropout rate
            rngs: Random number generators
        """
        self.window_size = window_size
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.scale = (embed_dim // n_heads) ** -0.5

        self.relative_position_bias_table = nnx.Param(
            jnp.zeros(((2 * window_size - 1) * (2 * window_size - 1), self.n_heads))
        )

        coords_h = jnp.arange(window_size)
        coords_w = jnp.arange(window_size)
        coords = jnp.stack(jnp.meshgrid(coords_h, coords_w), axis=0)
        coords_flatten = jnp.reshape(coords, (2, -1))
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = jnp.transpose(relative_coords, (1, 2, 0))
        relative_coords = relative_coords.at[:, :, 0].add(window_size - 1)
        relative_coords = relative_coords.at[:, :, 1].add(window_size - 1)
        relative_coords = relative_coords.at[:, :, 0].multiply(2 * window_size - 1)
        self.relative_position_index = jnp.sum(relative_coords, axis=-1)

        self.qkv = nnx.Linear(embed_dim, 3 * embed_dim, rngs=rngs)
        self.attn_drop = nnx.Dropout(rate=attn_dropout, rngs=rngs)
        self.proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.proj_drop = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray | None = None):
        """Forward pass of the layer.

        Args:
          x: Input array with the shape (B, N, C) where B is the batch size, N is the
             number of tokens, and C is the embedding dimension.
          mask: Mask array for attention. Defaults to None.

        Returns:
            Output array after applying window attention.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, C // self.n_heads)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.swapaxes(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].reshape(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        relative_position_bias = relative_position_bias.transpose(2, 0, 1)
        attn = attn + jnp.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(B // nW, nW, self.n_heads, N, N)
            attn = attn + jnp.expand_dims(jnp.expand_dims(mask, axis=1), axis=0)
            attn = attn.reshape(-1, self.n_heads, N, N)
            attn = nnx.softmax(attn)
        else:
            attn = nnx.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).swapaxes(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DropPath(nnx.Module):
    """Drops Stochastic Depth per sample in main paths of residual networks.

    This module randomly drops entire layers (i.e., the entire residual block) during
    training to prevent overfitting. The drop probability is linearly increased from 0
    to the specified dropout rate. This improves generalization and reduces the need for
    hyperparameter tuning.

    Adapted from TIMM:
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    """

    def __init__(self, dropout: float = 0.0, scale=True):
        """Initializes the DropPath module.

        Args:
          dropout: Dropout rate for the stochastic depth
          scale: Whether to scale the stochastic depth. Defaults to True.
        """
        self.dropout = dropout
        self.scale = scale

    def __call__(self, x: jnp.ndarray, train=True):
        """Forward pass of the module.

        Args:
          x: Input array
          train: Whether to run in training mode. Defaults to True.

        Returns:
            Output array after applying stochastic depth.
        """
        if self.dropout == 0.0 or not train:
            return x
        keep_prob = 1 - self.dropout
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        # TODO: use nnx.Rngs
        random_tensor = jax.random.bernoulli(jax.random.PRNGKey(0), keep_prob, shape)
        if keep_prob > 0.0 and self.scale:
            random_tensor = random_tensor.astype(x.dtype) / keep_prob
        return x * random_tensor


class Identity(nnx.Module):
    """Identity is a no-op module.

    This module is used to represent the identity function, which returns the input
    tensor as is. It is used to simplify the code and make it more readable.
    """

    def __call__(self, x: jnp.ndarray):
        """Forward pass of the module.

        Args:
            x: Input array

        Returns:
            Output array. Identical to the input array.
        """
        return x


class PatchEmbedding(nnx.Module):
    """Splits the image into patches and projects them into an embedding space.

    This module uses a convolutional layer to divide the input into non-overlapping
    patches and then projects each patch into a higher-dimensional embedding space.
    A layer normalization is applied to the embedded patches.
    """

    def __init__(
        self, input_features: int, embed_dim: int, patch_size: int, *, rngs: nnx.Rngs
    ):
        """Initializes the PatchEmbedding module.

        Args:
          input_features: Number of input features (channels)
          embed_dim: Dimension of the embedding
          patch_size: Size of the patch (height and width)
          rngs: Random number generators
        """
        self.embed_dim = embed_dim
        self.input_features = input_features
        self.conv = nnx.Conv(
            in_features=input_features,
            out_features=embed_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            rngs=rngs,
        )
        self.norm = nnx.LayerNorm(embed_dim, epsilon=1e-5, rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        """Forward pass of the module.

        Args:
            x: Input array with shape (B, H, W, C) where B is the batch size, H is the
               height, W is the width, and C is the number of channels.

        Returns:
            Output array with shape (B, N, embed_dim) where N is the number of patches.
        """
        B, _, _, _ = x.shape
        x = self.conv(x)
        x = x.reshape((B, -1, self.embed_dim))
        x = self.norm(x)
        return x


class SwinTransformerBlock(nnx.Module):
    """A transformer block with window attention and an MLP.

    This module implements a Swin Tranformer block which includes a window-based
    multi-head self-attention (W-MSA) and a feedforward network with optional
    shift-based window partitioning.
    """

    def __init__(
        self,
        embed_dim: int,
        input_resolution: tuple[int, int],
        n_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
        *,
        rngs: nnx.Rngs,
    ):
        """Initializes the Swin Transformer block.

        Args:
            embed_dim: Dimension of the embedding.
            input_resolution: Input resolution (height and width).
            n_heads: Number of attention heads.
            window_size: Size of the window (height and width).
            shift_size: Size of the shift for shift-based partitioning.
            mlp_ratio: Ratio of the hidden layer to the embedding.
            dropout: Dropout rate for the feedforward network.
            attn_dropout: Dropout rate for attention.
            rngs: Random number generators.
        """
        self.shift_size = shift_size
        self.input_resolution = input_resolution
        self.window_size = window_size

        self.norm1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.attn = WindowAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            window_size=window_size,
            attn_dropout=attn_dropout,
            dropout=dropout,
            rngs=rngs,
        )
        self.drop_path = DropPath(dropout) if dropout > 0.0 else Identity()
        self.norm2 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.mlp = nnx.Sequential(
            nnx.Linear(embed_dim, int(embed_dim * mlp_ratio), rngs=rngs),
            nnx.gelu,
            nnx.Dropout(rate=dropout, rngs=rngs),
            nnx.Linear(int(embed_dim * mlp_ratio), embed_dim, rngs=rngs),
            nnx.Dropout(rate=dropout, rngs=rngs),
        )

    def _create_attn_mask(
        self, shift_size: int, window_size: int, input_resolution: tuple[int, int]
    ):
        """Create the attention mask for window-based attention.

        Args:
            shift_size: Size of the shift for shift-based partitioning.
            window_size: Size of the window (height and width).
            input_resolution: Input resolution (height and width).

        Returns:
            Attention mask for window-based attention if shift_size > 0, otherwise None.
        """
        if self.shift_size > 0:
            H, W = input_resolution
            mask = jnp.zeros((1, H, W, 1))
            h_slices = (
                slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None),
            )
            w_slices = (
                slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None),
            )

            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask.at[:, h, w, :].set(count)
                    count += 1

            mask_windows = window_partition(mask, window_size)
            mask_windows = mask_windows.reshape(-1, window_size * window_size)
            attn_mask = jnp.expand_dims(mask_windows, axis=1) - jnp.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = jnp.where(attn_mask != 0.0, -100.0, attn_mask)
            attn_mask = jnp.where(attn_mask == 0.0, 0.0, attn_mask)
        else:
            attn_mask = None

        return attn_mask

    def __call__(self, x: jnp.ndarray):
        """Forward pass of the block.

        Args:
            x: Input array with shape (B, L, C) where B is the batch size, L is the
               sequence length, and C is the embedding dimension.

        Returns:
            Output array after applying the Swin Transformer block.
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has invalid size"

        input_resolution = min(H, W)
        if input_resolution <= self.window_size:
            shift_size = 0
            window_size = input_resolution
        else:
            shift_size = self.shift_size
            window_size = self.window_size

        shortcut = x
        x = self.norm1(x)
        x = x.reshape(B, H, W, C)

        # Cyclic shift
        if shift_size > 0:
            shifted_x = jnp.roll(x, shift=(-shift_size, -shift_size), axis=(1, 2))
        else:
            shifted_x = x

        windows = window_partition(shifted_x, window_size)
        windows = windows.reshape(-1, window_size * window_size, C)

        attn_windows = self.attn(
            windows,
            mask=self._create_attn_mask(shift_size, window_size, self.input_resolution),
        )

        # Merge windows
        attn_windows = attn_windows.reshape(-1, window_size, window_size, C)
        shifted_x = window_reverse(attn_windows, window_size, H, W)

        # Reverse the cyclic shift
        if shift_size > 0:
            x = jnp.roll(shifted_x, shift=(shift_size, shift_size), axis=(1, 2))
        else:
            x = shifted_x

        x = x.reshape(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # Feedforward
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        return x


class SwinLayer(nnx.Module):
    """A sequence of Swin Transformer blocks and an optional patch merging layer.

    This module implements a layer of the Swin Transformer, consisting of multiple
    Swin Transformer blocks followed by an optional patch merging layer.
    """

    def __init__(
        self,
        embed_dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        n_heads: int,
        window_size: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
        downsample=False,
        *,
        rngs: nnx.Rngs,
    ):
        """Initializes the Swin Layer.

        Args:
            embed_dim: Dimension of the embedding.
            input_resolution: Input resolution (height and width).
            depth: Number of Swin Transformer blocks in the layer.
            n_heads: Number of attention heads.
            window_size: Size of the window (height and width).
            mlp_ratio: Ratio of the hidden layer to the embedding.
            dropout: Dropout rate for the feedforward network.
            attn_dropout: Dropout rate for attention.
            downsample: Whether to apply patch merging. Defaults to False.
            rngs: Random number generators.
        """
        self.embed_dim = embed_dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.n_heads = n_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.attn_dropout = attn_dropout

        self.blocks = nnx.Sequential(
            *[
                SwinTransformerBlock(
                    embed_dim=embed_dim,
                    input_resolution=input_resolution,
                    n_heads=n_heads,
                    window_size=window_size,
                    shift_size=0 if (idx % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    rngs=rngs,
                )
                for idx in range(depth)
            ]
        )

        if downsample:
            self.downsample = PatchMerging(input_resolution, embed_dim, rngs=rngs)
        else:
            self.downsample = Identity()

    def __call__(self, x: jnp.ndarray):
        """Forward pass of the layer.

        Args:
            x: Input array with the shape (B, L, C) where B is the batch size, L is the
               sequence length, and C is the embedding dimension.

        Returns:
            Output array after applying the Swin Transformer layer.
        """
        x = self.blocks(x)
        x = self.downsample(x)
        return x


class SwinTransformer(nnx.Module):
    """Swin Transformer model.

    This module implements the Swin Transformer architecture, which consists of
    hierarchical feature representation with shifted windows for efficient and
    scalable self-attention computation.
    """

    def __init__(self, model_config: SwinTransformerConfig, *, rngs: nnx.Rngs):
        """Initializes the Swin Transformer model.

        Args:
          model_config: Configuration for the model.
          rngs: Random number generators.
        """
        self.config = model_config
        self.n_layers = len(self.config.depths)
        height, width, input_features = model_config.input_shape
        self.n_features = int(self.config.embed_dim * 2 ** (self.n_layers - 1))
        self.patches_resolution = (
            height // self.config.patch_size,
            width // self.config.patch_size,
        )

        self.n_patches = (height // model_config.patch_size) * (
            width // (model_config.patch_size)
        )
        self.patch_embed = PatchEmbedding(
            input_features=input_features,
            embed_dim=model_config.embed_dim,
            patch_size=model_config.patch_size,
            rngs=rngs,
        )
        self.pos_drop = nnx.Dropout(rate=model_config.dropout, rngs=rngs)
        self.pos_embed = nnx.Param(
            jnp.zeros((1, self.n_patches, model_config.embed_dim))
        )

        self.layers = []
        for layer_idx in range(self.n_layers):
            layer = SwinLayer(
                embed_dim=int(self.config.embed_dim * 2**layer_idx),
                input_resolution=(
                    self.patches_resolution[0] // (2**layer_idx),
                    self.patches_resolution[1] // (2**layer_idx),
                ),
                depth=self.config.depths[layer_idx],
                n_heads=self.config.n_heads[layer_idx],
                window_size=self.config.window_size,
                mlp_ratio=self.config.mlp_ratio,
                dropout=self.config.dropout,
                attn_dropout=self.config.attn_dropout,
                downsample=layer_idx < self.n_layers - 1,
                rngs=rngs,
            )
            self.layers.append(layer)

        self.norm = nnx.LayerNorm(self.n_features, rngs=rngs)
        self.pool = AdaptiveAvgPool1D(1)
        self.head = nnx.Linear(self.n_features, model_config.num_classes, rngs=rngs)

    def __call__(self, x: jnp.ndarray, train=True):
        """Forward pass of the model.

        Args:
          x: Input array with the shape (B, H, W, C) where B is the batch size, H is the
             height, W is the width, and C is the number of channels.
          train: Whether to run in training mode. Defaults to True.

        Returns:
            Output array after applying the Swin Transformer model.
        """
        ## Forward pass through the model
        x = self.patch_embed(x)
        if self.config.use_absolute_pos_embed:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.pool(x)

        # Flatten the tensor
        x = x.reshape(x.shape[0], -1)

        ## Classification head
        x = self.head(x)
        return x


if __name__ == "__main__":
    patch_datasets_warning()
    console.log("Loading dataset")
    batch_size = 4
    dataset = dummy_image_classification_dataset(
        batch_size=batch_size, shuffle=False, shape=(224, 224, 3)
    )
    input_shape = dataset.metadata.input_shape

    console.log(f"Dataset metadata: {dataset.metadata}")
    assert dataset.test is not None, "Test dataset is None"

    console.log("Configuring model")
    model_config = SwinTransformerConfig(
        num_classes=dataset.metadata.num_classes, input_shape=input_shape
    )
    model = SwinTransformer(model_config, rngs=nnx.Rngs(0))

    trainer_config = ImageClassificationParallelTrainerConfig(
        batch_size=batch_size, epochs=5
    )
    trainer = ImageClassificationParallelTrainer(model, trainer_config)
    trainer.train_and_evaluate(dataset.train, dataset.test)
