import torch
import torch.nn.functional as F
from scratch.utils.device import get_device
from scratch.utils.logging import setup_logger
from scratch.deep_learning.cnn import ConvNet
from tests.deep_learning.utils import compute_accuracy, load_mnist


batch_size = 128
num_epochs = 1
learning_rate = 0.1

random_seed = 1
torch.manual_seed(random_seed)

logger = setup_logger()

device = get_device()
logger.info(f"Using device: {device.type}")


def test_cnn():
    train_loader, test_loader = load_mnist(batch_size=batch_size)

    model = ConvNet(num_classes=10).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    model = model.train()
    for epoch in range(num_epochs):
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(device)
            targets = targets.to(device)

            logits, _ = model(features)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        model = model.eval()
        train_accuracy = compute_accuracy(model, train_loader, device)
        logger.info(
            f"Epoch: {epoch + 1}/{num_epochs} | Training Accuracy: {train_accuracy}"
        )
        assert train_accuracy.item() > 0.85
        assert train_accuracy.item() < 1.0

    with torch.set_grad_enabled(False):
        test_accuracy = compute_accuracy(model, test_loader, device)
        logger.info(f"Test Accuracy: {test_accuracy}")
        assert test_accuracy.item() > 0.85
        assert test_accuracy.item() < 1.0
