from scratch.deep_learning.cnn import CNN
from scratch.deep_learning.trainer.image_classification import (
    ImageClassificationTrainer,
)
from scratch.datasets.dataset import mnist_dataset


if __name__ == "__main__":
    model = CNN(num_classes=10)
    trainer = ImageClassificationTrainer(
        model=model, input_shape=(28, 28, 1), num_classes=10
    )
    dataset = mnist_dataset(batch_size=128, shuffle=True)
    trainer.train(dataset=dataset)
