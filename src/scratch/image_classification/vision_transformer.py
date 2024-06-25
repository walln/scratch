"""A simple vision transformer model."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from scratch.datasets.image_classification_dataset import (
    dummy_image_classification_dataset,
)
from scratch.image_classification.trainer import (
    ImageClassificationParallelTrainer,
    ImageClassificationParallelTrainerConfig,
)
from scratch.utils.logging import console


@dataclass
class VisionTransformerConfig:
    """Configuration for the VisionTransformer model."""

    input_shape: tuple[int, int, int] = (224, 224, 3)
    """Input shape of the model."""
    num_classes: int = 10
    """Number of classes in the dataset."""
    patch_size: int = 16
    """Size of the patch."""
    n_layers: int = 12
    """Number of layers."""
    embed_dim: int = 768
    """Dimension of the embedding."""
    hidden_dim: int = 3072
    """Dimension of the hidden layer."""
    n_heads: int = 12
    """Number of heads."""
    dropout: float = 0.1
    """Dropout rate."""


class Block(nnx.Module):
    """A transformer block."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        n_heads: int,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        """Initializes the transformer block.

        Args:
          embed_dim: Dimension of the embedding
          hidden_dim: Dimension of the hidden layer
          n_heads: Number of heads
          dropout_rate: Dropout rate
          rngs: Random number generators
        """
        self.ln1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.ln2 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(n_heads, embed_dim, rngs=rngs, decode=False)
        self.mlp = nnx.Sequential(
            nnx.Linear(embed_dim, hidden_dim, rngs=rngs),
            nnx.gelu,
            nnx.Dropout(rate=dropout_rate, rngs=rngs),
            nnx.Linear(hidden_dim, embed_dim, rngs=rngs),
            nnx.Dropout(rate=dropout_rate, rngs=rngs),
        )

    def __call__(self, x: jnp.ndarray):
        """Forward pass of the model.

        Args:
          x: Input array

        Returns:
          Output array
        """
        input_x = self.ln1(x)
        x = x + self.attn(input_x)
        x = x + self.mlp(self.ln2(x))
        return x


class VisionTransformer(nnx.Module):
    """A simple vision transformer model."""

    def __init__(self, model_config: VisionTransformerConfig, *, rngs: nnx.Rngs):
        """Initializes the simple vision transformer model.

        Args:
          model_config: Configuration for the model
          rngs: Random number generators
        """
        self.config = model_config
        height, width, _ = self.config.input_shape
        num_patches = (height // self.config.patch_size) * (
            width // self.config.patch_size
        )
        self.patch_embedding = nnx.Linear(
            model_config.input_shape[-1] * (model_config.patch_size**2),
            model_config.embed_dim,
            rngs=rngs,
        )
        self.transformer = nnx.Sequential(
            *[
                Block(
                    model_config.embed_dim,
                    model_config.hidden_dim,
                    model_config.n_heads,
                    dropout_rate=model_config.dropout,
                    rngs=rngs,
                )
                for _ in range(model_config.n_layers)
            ]
        )
        self.mlp_head = nnx.Sequential(
            nnx.LayerNorm(model_config.embed_dim, rngs=rngs),
            nnx.Linear(model_config.embed_dim, model_config.num_classes, rngs=rngs),
        )
        self.dropout = nnx.Dropout(rate=model_config.dropout, rngs=rngs)

        # Positional embeddings
        self.cls_token = nnx.Param(
            jax.random.normal(jax.random.PRNGKey(0), (1, 1, self.config.embed_dim))
        )
        self.pos_embedding = nnx.Param(
            jax.random.normal(
                jax.random.PRNGKey(1), (1, 1 + num_patches, self.config.embed_dim)
            )
        )

    def __call__(self, x: jnp.ndarray, train=True):
        """Forward pass of the model.

        Args:
          x: Input array
          train: Whether the model is in training mode

        Returns:
          Output array
        """

        def img_to_patch(x: jnp.ndarray, patch_size: int):
            # assuming x is of shape ( H, W, C)
            print(f"In img_to_patch: {x.shape}")
            B, H, W, C = x.shape
            x = x.reshape(
                B, H // patch_size, patch_size, W // patch_size, patch_size, C
            )
            x = jnp.swapaxes(x, 2, 3)  # Swap patch_size dimensions
            x = x.reshape(B, -1, patch_size * patch_size * C)
            return x

        x = img_to_patch(x, self.config.patch_size)
        print(f"X shape: {x.shape}")
        B, T, _ = x.shape
        x = self.patch_embedding(x)

        # Add CLS token and positional encoding
        cls_token = jnp.broadcast_to(
            self.cls_token.value, (B, 1, self.config.embed_dim)
        )
        x = jnp.concatenate([cls_token, x], axis=1)  # type: ignore - nnx.Param should be ArrayLike?
        x = x + self.pos_embedding[:, : T + 1]

        # Apply Transformer
        x = self.dropout(x)
        x = jnp.swapaxes(x, 0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out


if __name__ == "__main__":
    console.log("Loading dataset")
    batch_size = 16
    input_shape = (224, 224, 3)
    dataset = dummy_image_classification_dataset(
        batch_size=batch_size, shuffle=True, shape=input_shape, num_samples=128
    )

    console.log(f"Dataset metadata: {dataset.metadata}")
    assert dataset.test is not None, "Test dataset is None"

    console.log("Configuring model")
    model_config = VisionTransformerConfig(
        num_classes=dataset.metadata.num_classes, input_shape=input_shape
    )
    model = VisionTransformer(model_config, rngs=nnx.Rngs(0))

    trainer_config = ImageClassificationParallelTrainerConfig(
        batch_size=batch_size, epochs=1
    )
    trainer = ImageClassificationParallelTrainer(model, trainer_config)
    trainer.train_and_evaluate(dataset.train, dataset.test)
