"""CNN Model using NNX api from Flax."""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from rich.progress import MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn
from scratch.datasets.dataset import (
    DataLoader,
)
from scratch.datasets.image_classification_dataset import (
    ImageClassificationBatch,
    mnist_dataset,
)


@dataclass
class CNNConfig:
    """Configuration for the CNN model."""

    num_classes: int = 10


class CNN(nnx.Module):
    """A simple CNN model."""

    def __init__(self, config: CNNConfig, *, rngs: nnx.Rngs):
        """Initializes the simple CNN model.

        Args:
            config: Configuration for the model
            rngs: Random number generators
        """
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
        self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
        self.linear2 = nnx.Linear(256, config.num_classes, rngs=rngs)

    def __call__(self, x):
        """Forward pass of the model.

        Args:
            x: Input array

        Returns:
            Output array
        """
        x = self.avg_pool(nnx.relu(self.conv1(x)))
        x = self.avg_pool(nnx.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = nnx.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class TrainState(nnx.Optimizer):
    """Train state for the CNN model."""

    model: CNN
    tx: optax.GradientTransformation
    metrics: nnx.MultiMetric

    def __init__(self, model, tx, metrics: nnx.MultiMetric):
        """Initializes the train state."""
        self.metrics = metrics
        super().__init__(model, tx)

    def update(self, *, grads, **updates):
        """Updates the train state in-place."""
        self.metrics.update(**updates)
        super().update(grads)

    def update_metrics(self, **updates):
        """Updates the metrics in-place."""
        self.metrics.update(**updates)

    def reset_metrics(self):
        """Resets the metrics."""
        self.metrics.reset()

    def compute_metrics(self):
        """Realizes the metric values."""
        return self.metrics.compute()


# TODO: Mesh and SPMD parallelism
class CNNParallelTrainer:
    """Trains a CNN model using NNX and SPMD parallelism."""

    def __init__(
        self, model: CNN, batch_size: int = 64, learning_rate=0.005, momentum=0.9
    ):
        """Initializes the CNN trainer.

        Args:
            model: The CNN model to train
            batch_size: The batch size
            learning_rate: The learning rate
            momentum: The momentum
        """
        self.model = model
        self.batch_size = batch_size
        self.jit_train_step = nnx.jit(CNNParallelTrainer.train_step)
        self.jit_eval_step = nnx.jit(CNNParallelTrainer.eval_step)
        self.state = self._create_train_state(
            learning_rate=learning_rate, momentum=momentum
        )

    def _create_train_state(self, learning_rate: float, momentum: float):
        metrics = nnx.MultiMetric(
            accuracy=nnx.metrics.Accuracy(),
            loss=nnx.metrics.Average("loss"),
        )
        state = TrainState(self.model, optax.adamw(learning_rate, momentum), metrics)
        return state

    @staticmethod
    def train_step(
        model: CNN,
        train_state: TrainState,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
    ) -> jnp.ndarray:
        """Performs a single training step.

        Args:
            model: The model
            train_state: The training state
            inputs: The input data
            targets: The target data

        Returns:
            The loss
        """

        def loss_fn(model: CNN):
            # model.set_attributes(deterministic=False, decode=False)
            logits = model(inputs)
            assert logits.shape == targets.shape
            loss = optax.softmax_cross_entropy(logits=logits, labels=targets).mean()
            return loss, logits

        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(model)
        train_state.update(
            grads=grads, loss=loss, logits=logits, labels=jnp.argmax(targets, axis=-1)
        )

        return loss

    def train(self, train_loader: DataLoader[ImageClassificationBatch]):
        """Trains the model on the entire training dataset.

        Args:
            train_loader: The training data loader
        """
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Training", total=None)
            for batch in train_loader:
                inputs, targets = batch["image"], batch["label"]
                inputs, targets = (
                    inputs.numpy().astype(jnp.float32),
                    targets.numpy().astype(jnp.int32),
                )
                self.jit_train_step(self.model, self.state, inputs, targets)
                progress.update(task, advance=self.batch_size)
        for (
            metric,
            value,
        ) in self.state.compute_metrics().items():
            print(f"train_{metric}: {value}")
        self.state.reset_metrics()

    @staticmethod
    def eval_step(
        model: CNN,
        train_state: TrainState,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
    ):
        """Evaluates the model on a single batch.

        Args:
            model: The model
            train_state: The training state
            inputs: The input data
            targets: The target data
        """
        logits = model(inputs)
        train_state.update_metrics(
            loss=0.0, logits=logits, labels=jnp.argmax(targets, axis=-1)
        )

    def eval(self, test_loader: DataLoader[ImageClassificationBatch]):
        """Evaluates the model on the entire test dataset.

        Args:
            test_loader: The test data loader
        """
        for batch in test_loader:
            inputs, targets = batch["image"], batch["label"]
            inputs, targets = (
                inputs.numpy().astype(jnp.float32),
                targets.numpy().astype(jnp.int32),
            )
            self.jit_eval_step(self.model, self.state, inputs, targets)
            break
        for (
            metric,
            value,
        ) in self.state.compute_metrics().items():
            print(f"test_{metric}: {value}")
        self.state.reset_metrics()

    # TODO: Save and load methods


"""
Trains a simple CNN model on the MNIST dataset.

Running on my RTX 3080ti, and loading the dataset from a memmapped file, the training
only takes ~20 seconds for the full training split and reaches ~98% accuracy on the test
split.
"""
if __name__ == "__main__":
    print("Jax Backend:", jax.default_backend())

    model_config = CNNConfig()
    model = CNN(model_config, rngs=nnx.Rngs(0))

    print("Loading dataset...")
    batch_size = 64
    dataset = mnist_dataset(
        batch_size=batch_size,
        shuffle=True,
    )
    print(f"Dataset metadata: {dataset.metadata}")
    assert dataset.test is not None, "Test dataset is None"

    trainer = CNNParallelTrainer(model, batch_size=batch_size)
    print("Training")
    trainer.train(train_loader=dataset.train)
    print("Evaluating")
    trainer.eval(test_loader=dataset.test)
