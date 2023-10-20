from torch import nn
import torch.nn.functional as F


class BasicResidualBlock(nn.Module):
    # Static expansion rate
    # adapted from notes by Sebastian Raschka
    expansion = 1

    def __init__(self, input_planes, planes, stride=1, downsample=None):
        super(BasicResidualBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride
        self.conv_1 = nn.Conv2d(
            input_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm_1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm_2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv_1(x)
        out = self.norm_1(out)
        out = self.relu(out)
        out = self.conv_2(out)
        out = self.norm_2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        num_layers: int = 2,
        block_type: nn.Module = BasicResidualBlock,
        is_grey: bool = False,
    ):
        super(ResNet, self).__init__()
        # Image are in RGB format
        self.input_dim = 3
        if is_grey:
            # Images are greyscale
            self.input_dim = 1
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.block_type = block_type
        self.inner = 64

        self.conv_1 = nn.Conv2d(
            self.input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.batch_norm_1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.block_type, 64, self.num_layers)
        # Adjusted stride to slow down spatial dimension reduction
        self.layer2 = self._make_layer(self.block_type, 128, self.num_layers, stride=1)
        self.layer3 = self._make_layer(self.block_type, 256, self.num_layers, stride=2)
        # Adjusted the fc layer's dimensions
        self.fc = nn.Linear(4096, num_classes)

        # Intialize weights using Kaiming Normal and saturated batch norm
        # TODO: try this implenentation of weight initalization
        # apparently this works better for skip connections
        # https://arxiv.org/pdf/1709.02956v1.pdf
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks=2, stride=1):
        downsample = None
        if stride != 1 or self.inner != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inner,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inner, planes, stride, downsample))
        self.inner = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inner, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas
