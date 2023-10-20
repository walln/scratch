import torch
from scratch.utils.device import get_device
from scratch.utils.logging import setup_logger
from scratch.utils.reproducability import set_random_seed
from scratch.deep_learning.cnn.model import ConvNet


def test_cnn():
    set_random_seed()
    device = get_device()

    logger = setup_logger()
    logger.info(f"Using device: {device.type}")

    model = ConvNet(num_classes=10).to(device)
    X = torch.randn(1, 1, 28, 28).to(device)
    logits, _ = model(X)

    assert logits.shape == (1, 10)
