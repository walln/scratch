"""Resnet model implemented in Equinox."""

from functools import partial
from typing import Optional, Sequence, Tuple

import equinox as eqx
import jax
from jaxtyping import Array


class Block(eqx.nn.StatefulLayer):
    """Bottleneck block."""

    expansion: int
    out_channels: int
    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    conv2: eqx.nn.Conv2d
    bn2: eqx.nn.BatchNorm
    conv3: Optional[eqx.nn.Conv2d] = None
    bn3: Optional[eqx.nn.BatchNorm] = None
    downsample: Optional[eqx.nn.Sequential] = None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[eqx.nn.Sequential] = None,
        *,
        key: Optional[Array] = None,
    ):
        """Create a bottleneck block."""
        use_bottleneck = out_channels // in_channels >= 4
        self.expansion = 4 if use_bottleneck else 1
        mid_channels = (
            out_channels // self.expansion if use_bottleneck else out_channels
        )
        if key is None:
            raise ValueError("key must be provided")
        keys = jax.random.split(key, 3 if use_bottleneck else 2)

        if use_bottleneck:
            # Bottleneck architecture
            self.conv1 = eqx.nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                key=keys[0],
            )
            self.bn1 = eqx.nn.BatchNorm(input_size=mid_channels, axis_name="batch")
            self.conv2 = eqx.nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                key=keys[1],
            )
            self.bn2 = eqx.nn.BatchNorm(input_size=mid_channels, axis_name="batch")
            self.conv3 = eqx.nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                key=keys[2],
            )
            self.bn3 = eqx.nn.BatchNorm(input_size=out_channels, axis_name="batch")
        else:
            # Regular ResNet block (as in ResNet-18/34)
            self.conv1 = eqx.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                key=keys[0],
            )
            self.bn1 = eqx.nn.BatchNorm(input_size=out_channels, axis_name="batch")
            self.conv2 = eqx.nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                key=keys[1],
            )
            self.bn2 = eqx.nn.BatchNorm(input_size=out_channels, axis_name="batch")
            self.conv3 = None  # Not used in regular block

        self.downsample = downsample
        self.out_channels = out_channels

    def __call__(self, x, state, key) -> Tuple[Array, eqx.nn.State]:
        """Forward pass on bottleneck block."""
        residual = x

        out = self.conv1(x)
        out, state = self.bn1(out, state=state)
        out = jax.nn.relu(out)
        out = self.conv2(out)
        out, state = self.bn2(out, state=state)
        out = jax.nn.relu(out)

        if self.conv3 and self.bn3:
            out = self.conv3(out)
            out, state = self.bn3(out, state=state)
            out = jax.nn.relu(out)

        if self.downsample:
            residual, state = self.downsample(x, state=state)

        out += residual
        out = jax.nn.relu(out)
        return out, state


class ResNet(eqx.Module):
    """Resnet model implemented in Equinox."""

    in_channels: int
    conv_1: eqx.nn.Conv2d
    bn_1: eqx.nn.BatchNorm
    maxpool: eqx.nn.MaxPool2d
    layer_1: eqx.nn.Sequential
    layer_2: eqx.nn.Sequential
    layer_3: eqx.nn.Sequential
    layer_4: eqx.nn.Sequential
    # avgpool: eqx.nn.AvgPool2d
    avgpool: eqx.nn.AdaptiveAvgPool2d
    fc: eqx.nn.Linear

    def __init__(self, layers: Sequence[int], num_classes: int, key: Array):
        """Create a resnet model."""
        keys = jax.random.split(key, len(layers) + 1)
        self.in_channels = 64

        self.conv_1 = eqx.nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            key=keys[0],
        )
        self.bn_1 = eqx.nn.BatchNorm(input_size=64, axis_name="batch")
        self.maxpool = eqx.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer_1 = self._make_layer(
            out_channels=64, blocks=layers[0], stride=1, key=keys[1]
        )
        self.layer_2 = self._make_layer(
            out_channels=128, blocks=layers[1], stride=2, key=keys[2]
        )
        self.layer_3 = self._make_layer(
            out_channels=256, blocks=layers[2], stride=2, key=keys[3]
        )
        self.layer_4 = self._make_layer(
            out_channels=512, blocks=layers[3], stride=2, key=keys[4]
        )

        # self.avgpool = eqx.nn.AvgPool2d(kernel_size=7, stride=1)
        self.avgpool = eqx.nn.AdaptiveAvgPool2d(target_shape=(1, 1))
        self.fc = eqx.nn.Linear(in_features=512, out_features=num_classes, key=keys[4])

    def _make_layer(
        self,
        out_channels: int,
        blocks: int,
        stride: int,
        key: Array,
    ):
        keys = jax.random.split(key, blocks + 2)

        first_block = Block(
            in_channels=self.in_channels,
            out_channels=out_channels,
            stride=stride,
            downsample=None,  # Temporarily None, to be potentially updated
            key=keys[0],
        )

        if stride != 1 or self.in_channels != out_channels * first_block.expansion:
            downsample = eqx.nn.Sequential(
                [
                    eqx.nn.Conv2d(
                        in_channels=self.in_channels,
                        out_channels=out_channels * first_block.expansion,
                        kernel_size=1,
                        stride=stride,
                        key=keys[1],
                    ),
                    eqx.nn.BatchNorm(
                        input_size=out_channels * first_block.expansion,
                        axis_name="batch",
                    ),
                ]
            )

            # Update the downsample function of the first block
            first_block = Block(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=stride,
                downsample=downsample,
                key=keys[0],
            )

        layers = []
        layers.append(first_block)
        self.in_channels = out_channels * first_block.expansion
        for i in range(1, blocks):
            layers.append(
                Block(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    key=keys[i + 1],
                )
            )

        return eqx.nn.Sequential(layers)

    def __call__(self, x: Array, state: eqx.nn.State) -> Tuple[Array, eqx.nn.State]:
        """Run a forward pass on resnet model."""
        x = x.transpose(2, 1, 0)
        x = self.conv_1(x)
        x, state = self.bn_1(x, state=state)
        x = jax.nn.relu(x)
        x = self.maxpool(x)
        x, state = self.layer_1(x, state=state)
        x, state = self.layer_2(x, state=state)
        x, state = self.layer_3(x, state=state)
        x, state = self.layer_4(x, state=state)
        x = self.avgpool(x)
        x = x.ravel()
        x = self.fc(x)
        return x, state


ResNet18 = partial(ResNet, layers=[2, 2, 2, 2])
ResNet34 = partial(ResNet, layers=[3, 4, 6, 3])
ResNet50 = partial(ResNet, layers=[3, 4, 6, 3])
