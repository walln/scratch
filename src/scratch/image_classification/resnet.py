"""ResNet model."""

from functools import partial
from typing import Literal

from flax import nnx
from scratch.datasets.image_classification_dataset import (
    patch_datasets_warning,
    tiny_imagenet_dataset,
)
from scratch.image_classification.trainer import (
    ImageClassificationParallelTrainer,
    ImageClassificationParallelTrainerConfig,
)
from scratch.utils.logging import console


class ResNetConfig:
    """Configuration for a ResNet model."""

    def __init__(
        self,
        num_classes: int,
        input_channels: int,
        depths: list[int],
        bottleneck=False,
    ):
        """Initializes the ResNet configuration.

        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels
            depths: List of depths for each stage
            bottleneck: Whether to use bottleneck blocks
        """
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.depths = depths  # List of depths for each stage
        self.bottleneck = bottleneck  # Whether to use bottleneck blocks

    @classmethod
    def from_preset(
        cls,
        preset_name: Literal[
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
        ],
        num_classes: int,
        input_channels: int = 3,
    ):
        """Creates a ResNetConfig from a preset.

        Args:
            preset_name: Name of the preset (e.g., 'resnet18')
            num_classes: Number of output classes
            input_channels: Number of input channels

        Returns:
            A ResNetConfig instance with the preset configuration
        """
        PRESETS = {
            "resnet18": {"depths": [2, 2, 2, 2], "bottleneck": False},
            "resnet34": {"depths": [3, 4, 6, 3], "bottleneck": False},
            "resnet50": {"depths": [3, 4, 6, 3], "bottleneck": True},
            "resnet101": {"depths": [3, 4, 23, 3], "bottleneck": True},
            "resnet152": {"depths": [3, 8, 36, 3], "bottleneck": True},
        }
        if preset_name not in PRESETS:
            raise ValueError(f"Unknown preset name: {preset_name}")
        config = PRESETS[preset_name]
        return cls(num_classes=num_classes, input_channels=input_channels, **config)


class ResNetBlock(nnx.Module):
    """A standard ResNet block."""

    def __init__(self, in_channels, out_channels, *, strides=(1, 1), rngs: nnx.Rngs):
        """Initializes the ResNet block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            strides: Strides for the first convolutional layer
            rngs: Random number generators
        """
        self.conv1 = nnx.Conv(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            strides=strides,
            padding="same",
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            out_channels, out_channels, kernel_size=(3, 3), padding="same", rngs=rngs
        )

        if in_channels != out_channels or strides != (1, 1):
            self.shortcut = nnx.Conv(
                in_channels,
                out_channels,
                kernel_size=(1, 1),
                strides=strides,
                rngs=rngs,
            )
        else:
            self.shortcut = lambda x: x

    def __call__(self, x):
        """Forward pass of the ResNet block.

        Args:
            x: Input array

        Returns:
            Output array
        """
        shortcut = self.shortcut(x)
        x = nnx.relu(self.conv1(x))
        x = self.conv2(x)
        return nnx.relu(x + shortcut)


class BottleneckResNetBlock(nnx.Module):
    """A bottleneck ResNet block."""

    def __init__(self, in_channels, out_channels, *, strides=(1, 1), rngs: nnx.Rngs):
        """Initializes the bottleneck ResNet block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            strides: Strides for the first convolutional layer
            rngs: Random number generators
        """
        bottleneck_channels = out_channels // 4
        self.conv1 = nnx.Conv(
            in_channels, bottleneck_channels, kernel_size=(1, 1), rngs=rngs
        )
        self.conv2 = nnx.Conv(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=(3, 3),
            strides=strides,
            padding="same",
            rngs=rngs,
        )
        self.conv3 = nnx.Conv(
            bottleneck_channels, out_channels, kernel_size=(1, 1), rngs=rngs
        )

        if in_channels != out_channels or strides != (1, 1):
            self.shortcut = nnx.Conv(
                in_channels,
                out_channels,
                kernel_size=(1, 1),
                strides=strides,
                rngs=rngs,
            )
        else:
            self.shortcut = lambda x: x

    def __call__(self, x):
        """Forward pass of the bottleneck ResNet block.

        Args:
            x: Input array

        Returns:
            Output array
        """
        shortcut = self.shortcut(x)
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = self.conv3(x)
        return nnx.relu(x + shortcut)


class ResNet(nnx.Module):
    """A configurable ResNet model."""

    def __init__(self, config: ResNetConfig, *, rngs: nnx.Rngs):
        """Initializes the configurable ResNet model.

        Args:
            config: Configuration for the model
            rngs: Random number generators
        """
        self.config = config
        self.conv1 = nnx.Conv(
            self.config.input_channels,
            64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding="same",
            rngs=rngs,
        )
        self.max_pool = partial(
            nnx.max_pool, window_shape=(3, 3), strides=(2, 2), padding="same"
        )

        self.layer1 = self._make_layer(64, 64, config.depths[0], rngs)
        self.layer2 = self._make_layer(64, 128, config.depths[1], rngs, strides=(2, 2))
        self.layer3 = self._make_layer(128, 256, config.depths[2], rngs, strides=(2, 2))
        self.layer4 = self._make_layer(256, 512, config.depths[3], rngs, strides=(2, 2))

        self.global_avg_pool = lambda x: x.mean(axis=(1, 2))
        self.fc = nnx.Linear(512, config.num_classes, rngs=rngs)

    def _make_layer(self, in_channels, out_channels, blocks, rngs, strides=(1, 1)):
        """Creates a ResNet layer composed of multiple ResNet blocks.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            blocks: Number of blocks in the layer
            rngs: Random number generators
            strides: Strides for the first block in the layer

        Returns:
            A ResNet layer
        """
        layers = []
        for i in range(blocks):
            if i == 0:
                layers.append(
                    self._make_block(
                        in_channels, out_channels, strides=strides, rngs=rngs
                    )
                )
            else:
                layers.append(self._make_block(out_channels, out_channels, rngs=rngs))
        return nnx.Sequential(*layers)

    def _make_block(self, in_channels, out_channels, strides=(1, 1), *, rngs: nnx.Rngs):
        """Creates a ResNet block or bottleneck block based on the configuration.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            strides: Strides for the first convolutional layer
            rngs: Random number generators

        Returns:
            A ResNet block or bottleneck block
        """
        if self.config.bottleneck:
            return BottleneckResNetBlock(
                in_channels, out_channels, strides=strides, rngs=rngs
            )
        else:
            return ResNetBlock(in_channels, out_channels, strides=strides, rngs=rngs)

    def __call__(self, x):
        """Forward pass of the model.

        Args:
            x: Input array

        Returns:
            Output array
        """
        x = nnx.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_avg_pool(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    patch_datasets_warning()
    console.log("Loading dataset")
    batch_size = 32
    dataset = tiny_imagenet_dataset(
        batch_size=batch_size,
    )

    console.log(f"Dataset metadata: {dataset.metadata}")
    assert dataset.test is not None, "Test dataset is None"

    console.log("Configuring model")
    model_config = ResNetConfig.from_preset(
        "resnet18", num_classes=dataset.metadata.num_classes
    )
    model = ResNet(model_config, rngs=nnx.Rngs(0))

    trainer_config = ImageClassificationParallelTrainerConfig(
        batch_size=batch_size, learning_rate=0.01, epochs=3
    )
    trainer = ImageClassificationParallelTrainer(model, trainer_config)
    trainer.train_and_evaluate(dataset.train, dataset.test)
