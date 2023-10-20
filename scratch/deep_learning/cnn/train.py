from loguru import logger
import torch
import torch.nn.functional as F
from scratch.deep_learning.cnn.model import ConvNet
from scratch.deep_learning.trainer import Trainer
from scratch.utils.dataset import load_mnist
from scratch.utils.device import get_device
from scratch.utils.evaluation import compute_accuracy
from scratch.utils.reproducability import set_random_seed


class CNNTrainer(Trainer):
    def __init__(
        self,
        batch_size: int = 128,
        num_epochs: int = 1,
        learning_rate: float = 0.1,
        random_seed: int = 1,
    ):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.random_seed = random_seed

        set_random_seed(self.random_seed)

        self.device = get_device()

    def train(self):
        logger.debug(f"Using device: {self.device.type}")
        train_loader, test_loader = load_mnist(batch_size=self.batch_size)
        logger.debug(f"Loaded dataloaders with batch size: {self.batch_size}")

        model = ConvNet(num_classes=10).to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)

        model = model.train()
        logger.debug(f"Training model for {self.num_epochs} epochs")
        for epoch in range(self.num_epochs):
            for _, (features, targets) in enumerate(train_loader):
                features = features.to(self.device)
                targets = targets.to(self.device)

                logits, _ = model(features)

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model = model.eval()

            train_accuracy = compute_accuracy(model, train_loader, self.device)
            logger.debug(
                f"Epoch: {epoch + 1}/{self.num_epochs} | Training Accuracy: {train_accuracy}"
            )

            assert train_accuracy.item() > 0.85
            assert train_accuracy.item() < 1.0

        with torch.set_grad_enabled(False):
            test_accuracy = compute_accuracy(model, test_loader, self.device)
            logger.debug(f"Test Accuracy: {test_accuracy}")
            assert test_accuracy.item() > 0.85
            assert test_accuracy.item() < 1.0


if __name__ == "__main__":
    trainer = CNNTrainer()
    trainer.train()
