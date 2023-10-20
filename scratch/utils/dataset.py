from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


def load_mnist(batch_size: int = 128):
    train_dataset = datasets.MNIST(
        root="data", train=True, transform=transforms.ToTensor(), download=True
    )

    test_dataset = datasets.MNIST(
        root="data", train=False, transform=transforms.ToTensor()
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
