from torch import nn
import torch.nn.functional as F

# TODO: Currently only supports 28x28 images for MNIST
# Update to support dynamic image sizes


class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=1
        )

        self.pool_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.conv_2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1
        )

        self.pool_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)

        self.linear_1 = nn.Linear(49 * 16, num_classes)

        # Intialize weights with Xavier Uniform and zero bias
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()
                if m.bias is not None:
                    m.bias.detach().zero_()

    def forward(self, x):
        out = self.conv_1(x)
        out = F.relu(out)
        out = self.pool_1(out)

        out = self.conv_2(out)
        out = F.relu(out)
        out = self.pool_2(out)

        logits = self.linear_1(out.view(-1, 7 * 7 * 16))
        probas = F.softmax(logits, dim=1)
        return logits, probas
