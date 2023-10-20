import torch
from scratch.utils.device import get_device
from scratch.utils.logging import setup_logger
from scratch.deep_learning.resnet18.model import ResNet
from scratch.utils.reproducability import set_random_seed


def test_resnet18():
    set_random_seed()
    device = get_device()

    logger = setup_logger()
    logger.info(f"Using device: {device.type}")

    model = ResNet(num_classes=10, is_grey=True).to(device)
    X = torch.randn(1, 1, 28, 28).to(device)
    logits, _ = model(X)

    model = ResNet(num_classes=10, is_grey=False).to(device)
    X = torch.randn(1, 3, 28, 28).to(device)
    logits, _ = model(X)

    assert logits.shape == (1, 10)
