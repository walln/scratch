from scratch.deep_learning.trainer import ImageClassificationTrainer
from scratch.deep_learning.resnet import ResNet18
from scratch.datasets import mnist_dataset


if __name__ == "__main__":
    model = ResNet18(num_classes=10)
    trainer = ImageClassificationTrainer(
        model=model, input_shape=(28, 28, 1), num_classes=10
    )
    dataset = mnist_dataset(batch_size=128, shuffle=True)
    trainer.train(dataset=dataset)
