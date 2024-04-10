"""A simple CNN model for image classification using pytorch."""

from torch import Tensor, nn
from torch.nn import functional as F


class CNN(nn.Module):
    """CNN model for image classification."""

    def __init__(self, num_classes: int):
        """Initialize the model.

        Args:
        ----
            num_classes: the number of classes
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
        ----
            x: the input image
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)
        x = x.reshape((-1, x.shape[-3] * x.shape[-2] * x.shape[-1]))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
