"""ResNet v1.5 model implementation. Adapted from torchvision.models.resnet."""
from typing import Any, Callable, List, Optional, Sequence, Type, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array


def _conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, key=None):
    """Construct a 3x3 convolution layer with padding.

    Args:
    ----
        in_planes: number of input channels
        out_planes: number of output channels
        stride: stride of convolution
        groups: number of groups
        dilation: dilation of convolution
        key: random key for initialization

    Returns:
    -------
        Two-dimensional convolutional layer with 3x3 kernel size, padding, and
        specified stride, groups, and dilation.
    """
    return eqx.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        use_bias=False,
        dilation=dilation,
        key=key,
    )


def _conv1x1(in_planes, out_planes, stride=1, key=None):
    """Construct a 1x1 convolution layer.

    Args:
    ----
        in_planes: number of input channels
        out_planes: number of output channels
        stride: stride of convolution
        key: random key for initialization

    Returns:
    -------
        Two-dimensional convolutional layer with 1x1 kernel size and specified
        stride.
    """
    return eqx.nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, use_bias=False, key=key
    )


class ResNetBlock(eqx.Module):
    """Basic ResNet v1.5 block without bottleneck."""

    expansion: int
    conv_1: eqx.Module
    batch_norm_1: eqx.Module
    conv_2: eqx.Module
    batch_norm_2: eqx.Module
    relu: Callable[[jnp.ndarray], jnp.ndarray]
    downsample: eqx.Module
    stride: int

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        key=None,
    ):
        """Construct a basic ResNet block without bottleneck.

        Args:
        ----
            inplanes: number of input channels
            planes: number of output channels
            stride: stride of convolution
            downsample: downsampling layer
            groups: number of groups
            base_width: base width
            dilation: dilation of convolution
            norm_layer: normalization layer
            key: random key for initialization

        Raises:
        ------
            ValueError: if groups != 1 or base_width != 64
            NotImplementedError: if dilation > 1
        """
        super(ResNetBlock, self).__init__()

        if norm_layer is None:
            norm_layer = eqx.nn.BatchNorm
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # self.conv1 and self.downsample layers downsample the input when stride != 1
        keys = jax.random.split(key, 2)
        self.expansion = 1
        self.conv_1 = _conv3x3(inplanes, planes, stride, key=keys[0])
        self.batch_norm_1 = norm_layer(planes, axis_name="batch")
        self.conv_2 = _conv3x3(planes, planes, key=keys[1])
        self.batch_norm_2 = norm_layer(planes, axis_name="batch")
        self.relu = jax.nn.relu

        if downsample:
            self.downsample = downsample
        else:
            self.downsample = eqx.nn.Identity()

        self.stride = stride

    def __call__(self, x: Array, *, key: Optional[jax.random.PRNGKey] = None):
        """Forward pass through the block."""
        out = self.conv_1(x)
        out = self.batch_norm_1(out)
        out = self.relu(out)

        out = self.conv_2(out)
        out = self.batch_norm_2(out)

        identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class ResNetBottleneckBlock(eqx.Module):
    """Basic ResNet v1.5 block with bottleneck."""

    expansion: int
    conv_1: eqx.Module
    batch_norm_1: eqx.Module
    conv_2: eqx.Module
    batch_norm_2: eqx.Module
    conv_3: eqx.Module
    batch_norm_3: eqx.Module
    relu: Callable[[jnp.ndarray], jnp.ndarray]
    downsample: eqx.Module
    stride: int

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        key=None,
    ):
        """Construct a basic ResNet block with bottleneck.

        Args:
        ----
            inplanes: number of input channels
            planes: number of output channels
            stride: stride of convolution
            downsample: downsampling layer
            groups: number of groups
            base_width: base width
            dilation: dilation of convolution
            norm_layer: normalization layer
            key: random key for initialization
        """
        super(ResNetBottleneckBlock, self).__init__()

        if norm_layer is None:
            norm_layer = eqx.nn.BatchNorm

        self.expansion = 4
        width = int(planes * (base_width / 64.0)) * groups

        keys = jax.random.split(key, 3)

        # conv_1 and downsample layers downsample the input when stride != 1
        self.conv_1 = _conv1x1(inplanes, width, key=keys[0])
        self.batch_norm_1 = norm_layer(width, axis_name="batch")
        self.conv_2 = _conv3x3(width, width, stride, groups, dilation, key=keys[1])
        self.batch_norm_2 = norm_layer(width, axis_name="batch")
        self.conv_3 = _conv1x1(width, planes * self.expansion, key=keys[2])
        self.batch_norm_3 = norm_layer(planes * self.expansion, axis_name="batch")
        self.relu = jax.nn.relu

        if downsample:
            self.downsample = downsample
        else:
            self.downsample = eqx.nn.Identity()

        self.stride = stride

    def __call__(self, x: Array, *, key: Optional[jax.random.PRNGKey] = None):
        """Forward pass through the block."""
        out = self.conv_1(x)
        out = self.batch_norm_1(out)
        out = self.relu(out)

        out = self.conv_2(out)
        out = self.batch_norm_2(out)
        out = self.relu(out)

        out = self.conv_3(out)
        out = self.batch_norm_3(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


EXPANSIONS = {ResNetBlock: 1, ResNetBottleneckBlock: 4}
BlockType = Type[Union[ResNetBlock, ResNetBottleneckBlock]]


class ResNet(eqx.Module):
    """ResNet v1.5 model.

    Adapted from torchvision.models.resnet.
    """

    inplanes: int
    dilation: int
    groups: Sequence[int]
    base_width: int

    conv_1: eqx.Module
    batch_norm_1: eqx.Module
    relu: jax.nn.relu
    maxpool: eqx.Module
    layer_1: eqx.Module
    layer_2: eqx.Module
    layer_3: eqx.Module
    layer_4: eqx.Module
    avgpool: eqx.Module
    fc: eqx.Module

    def __init__(
        self,
        block_type: Type[BlockType],
        layers: List[int],
        num_classes=10,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: List[bool] = None,
        norm_layer: Any = None,
        *,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        """Construct a ResNet v1.5 model.

        Args:
        ----
            block_type: type of block to use
            layers: number of blocks in each layer
            num_classes: number of classes
            groups: number of groups
            width_per_group: width per group
            replace_stride_with_dilation: replace stride with dilation
            norm_layer: normalization layer
            key: random key for initialization

        """
        super(ResNet, self).__init__()

        if not norm_layer:
            norm_layer = eqx.nn.BatchNorm

        if norm_layer != eqx.nn.BatchNorm:
            raise NotImplementedError("Unsupported norm_layer")

        if key is None:
            key = jax.random.PRNGKey(0)

        keys = jax.random.split(key, 6)

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )

        self.conv_1 = eqx.nn.Conv2d(
            3,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            use_bias=False,
            key=keys[0],
        )

        self.batch_norm_1 = norm_layer(input_size=self.inplanes, axis_name="batch")

        self.relu = jax.nn.relu
        self.maxpool = eqx.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer_1 = self._make_layer(
            block_type, 64, layers[0], norm_layer, key=keys[1]
        )
        self.layer_2 = self._make_layer(
            block_type,
            128,
            layers[1],
            norm_layer,
            stride=2,
            dilate=replace_stride_with_dilation[0],
            key=keys[2],
        )
        self.layer_3 = self._make_layer(
            block_type,
            256,
            layers[2],
            norm_layer,
            stride=2,
            dilate=replace_stride_with_dilation[1],
            key=keys[3],
        )
        self.layer_4 = self._make_layer(
            block_type,
            512,
            layers[3],
            norm_layer,
            stride=2,
            dilate=replace_stride_with_dilation[2],
            key=keys[4],
        )

        self.avgpool = eqx.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = eqx.nn.Linear(512 * EXPANSIONS[block_type], num_classes, key=keys[5])

    def _make_layer(
        self,
        block_type: BlockType,
        planes: int,
        blocks: int,
        norm_layer: Any,
        stride=1,
        dilate=False,
        key=None,
    ):
        """Construct a ResNet layer.

        Args:
        ----
            block_type: block type to use
            planes: number of output channels
            blocks: number of blocks in the layer
            norm_layer: normalization layer
            stride: stride of convolution
            dilate: whether to dilate convolution
            key: random key for initialization

        Returns:
        -------
            ResNet layer
        """
        if dilate:
            self.dilation *= stride
            stride = 1

        keys = jax.random.split(key, blocks + 1)
        downsample = None

        if stride != 1 or self.inplanes != planes * EXPANSIONS[block_type]:
            downsample = eqx.nn.Sequential(
                [
                    _conv1x1(
                        self.inplanes,
                        planes * EXPANSIONS[block_type],
                        stride=stride,
                        key=keys[0],
                    ),
                    norm_layer(
                        input_size=planes * EXPANSIONS[block_type], axis_name="batch"
                    ),
                ]
            )

        prev_dilation = self.dilation
        layers = [
            block_type(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                prev_dilation,
                norm_layer,
                key=keys[1],
            )
        ]

        self.inplanes = planes * EXPANSIONS[block_type]
        for block_idx in range(1, blocks):
            layers.append(
                block_type(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    key=keys[block_idx + 1],
                )
            )

        return eqx.nn.Sequential(layers)

    def __call__(self, x: Array, *, key: jax.random.PRNGKey) -> Array:
        """Forward pass through the model.

        Args:
        ----
            x: input array (batchless with 3 channels)
            key: random key for initialization

        Returns:
        -------
            Output of the model.

        Raises:
        ------
            ValueError: if key is not provided.
        """
        if key is None:
            raise ValueError("key must be provided")

        keys = jax.random.split(key, 6)

        x = self.conv_1(x, key=keys[0])
        x = self.batch_norm_1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer_1(x, key=keys[1])
        x = self.layer_2(x, key=keys[2])
        x = self.layer_3(x, key=keys[3])
        x = self.layer_4(x, key=keys[4])

        x = self.avgpool(x)
        x = jnp.ravel(x)
        x = self.fc(x, key=keys[5])

        return x


def resnet18(**kwargs) -> ResNet:
    """Create a ResNet-18 model.

    (Deep Residual Learning for Image Recognition)[https://arxiv.org/pdf/1512.03385.pdf]

    Args:
    ----
        kwargs: keyword arguments for `ResNet`
    """
    model = ResNet(ResNetBlock, [2, 2, 2, 2], **kwargs)
    return model
