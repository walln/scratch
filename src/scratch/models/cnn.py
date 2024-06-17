"""CNN Model using NNX api from Flax."""

from dataclasses import dataclass
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from rich.console import Console
from rich.progress import Progress
from scratch.datasets.dataset import (
    DataLoader,
)
from scratch.datasets.image_classification_dataset import (
    ImageClassificationBatch,
    mnist_dataset,
)
from scratch.utils.logging import console, get_progress_widgets
from scratch.utils.timer import capture_time


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


@dataclass
class CNNParallelTrainerConfig:
    """Configuration for the CNNParallelTrainer."""

    batch_size: int = 64
    """The global batch size to be sharded across all devices."""
    learning_rate: float = 0.005
    """The learning rate for the optimizer."""
    momentum: float = 0.9
    """The momentum for the optimizer."""
    epochs: int = 2
    """The number of epochs to train for."""
    console: Console = console
    """The console to use for logging."""


# TODO: Mesh and SPMD parallelism
class CNNParallelTrainer:
    """Trains a CNN model using NNX and SPMD parallelism."""

    def __init__(self, model: CNN, trainer_config: CNNParallelTrainerConfig):
        """Initializes the CNN trainer.

        Args:
            model: The CNN model to train
            trainer_config: The configuration for the trainer
        """
        self.console = console
        self.model = model
        self.train_config = trainer_config
        self.jit_train_step = nnx.jit(CNNParallelTrainer.train_step)
        self.jit_eval_step = nnx.jit(CNNParallelTrainer.eval_step)
        self.state = self._create_train_state(
            learning_rate=trainer_config.learning_rate, momentum=trainer_config.momentum
        )
        self.mesh = Mesh(jax.devices(), ("x",))

    def _create_train_state(self, learning_rate: float, momentum: float):
        metrics = nnx.MultiMetric(
            accuracy=nnx.metrics.Accuracy(),
            loss=nnx.metrics.Average("loss"),
        )
        state = TrainState(self.model, optax.adamw(learning_rate, momentum), metrics)
        return state

    def train_and_evaluate(self, train_loader, test_loader):
        """Trains and evaluates the model for the specified number of epochs.

        Args:
            train_loader: The training data loader
            test_loader: The test data loader
        """
        self.console.log(f"Running on {jax.default_backend()} backend")
        self.console.log(f"Using {jax.device_count()} devices")
        self.console.log("Beginning training and evaluation")

        for epoch in range(self.train_config.epochs):
            console.rule(f"Epoch {epoch + 1}/{self.train_config.epochs}")
            with Progress(
                *get_progress_widgets(), console=self.console, transient=True
            ) as progress:
                with capture_time() as train_time:
                    self.train(train_loader, progress=progress)
                with capture_time() as eval_time:
                    self.eval(test_loader, progress=progress)
            console.log(f"train_time: {train_time():.2f}s")
            console.log(f"eval_time: {eval_time():.2f}s")
            console.log(f"total_time: {train_time() + eval_time():.2f}s")

    def log_metrics(self, step_type: Literal["train", "eval"], progress: Progress):
        """Logs the metrics from the training state and resets them.

        Args:
            step_type: The type of step
            progress: The progress manager
        """
        for (
            metric,
            value,
        ) in self.state.compute_metrics().items():
            if step_type == "eval" and metric == "loss":
                continue

            progress.log(f"{step_type}_{metric}: {value}")
        self.state.reset_metrics()

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

    def train(
        self,
        train_loader: DataLoader[ImageClassificationBatch],
        *,
        progress: Progress | None = None,
    ):
        """Trains the model on the entire training dataset.

        Args:
            train_loader: The training data loader
            progress: The progress manager
        """
        with self.mesh:
            # Define named sharding
            input_sharding = NamedSharding(self.mesh, PartitionSpec("x"))
            target_sharding = NamedSharding(self.mesh, PartitionSpec("x"))

            with (
                progress
                if progress
                else Progress(
                    *get_progress_widgets(), console=self.console, transient=True
                ) as progress
            ):
                task = progress.add_task(
                    "Training",
                    total=len(train_loader) * self.train_config.batch_size
                    if len(train_loader)
                    else None,
                )
                for batch in train_loader:
                    inputs, targets = batch["image"], batch["label"]
                    inputs, targets = (
                        jax.device_put(
                            inputs.numpy().astype(jnp.float32), input_sharding
                        ),
                        jax.device_put(
                            targets.numpy().astype(jnp.int32), target_sharding
                        ),
                    )
                    self.jit_train_step(self.model, self.state, inputs, targets)
                    progress.update(task, advance=self.train_config.batch_size)
            self.log_metrics("train", progress)

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

    def eval(
        self,
        test_loader: DataLoader[ImageClassificationBatch],
        *,
        progress: Progress | None = None,
    ):
        """Evaluates the model on the entire test dataset.

        Args:
            test_loader: The test data loader
            progress: The progress manager
        """
        with self.mesh:
            # Define named sharding
            input_sharding = NamedSharding(self.mesh, PartitionSpec("x"))
            target_sharding = NamedSharding(self.mesh, PartitionSpec("x"))

            with (
                progress
                if progress
                else Progress(
                    *get_progress_widgets(), console=self.console, transient=True
                ) as progress
            ):
                task = progress.add_task(
                    "Evaluating",
                    total=len(test_loader) * self.train_config.batch_size
                    if len(test_loader)
                    else None,
                )
                for batch in test_loader:
                    inputs, targets = batch["image"], batch["label"]
                    inputs, targets = (
                        jax.device_put(
                            inputs.numpy().astype(jnp.float32), input_sharding
                        ),
                        jax.device_put(
                            targets.numpy().astype(jnp.int32), target_sharding
                        ),
                    )
                    self.jit_eval_step(self.model, self.state, inputs, targets)
                    progress.update(task, advance=self.train_config.batch_size)
            self.log_metrics("eval", progress)

    # TODO: Save and load methods


"""
Trains a simple CNN model on the MNIST dataset.

Running on my RTX 3080ti, and loading the dataset from a memmapped file, the training
only takes ~20 seconds for an epoch the full training split and reaches ~98% accuracy on
the test split.

Will likely be even faster on a TPU or a multi-GPU setup. The trainer naturally supports
SPMD parallelism and can be easily adapted to use multiple devices with no changes.

The simple CNN model is too simple for decent MFU and the bottleneck is the data loading
"""
if __name__ == "__main__":
    console.log("Configuring model")
    model_config = CNNConfig()
    model = CNN(model_config, rngs=nnx.Rngs(0))

    console.log("Loading dataset")
    batch_size = 64
    dataset = mnist_dataset(
        batch_size=batch_size,
        shuffle=True,
    )

    console.log(f"Dataset metadata: {dataset.metadata}")
    assert dataset.test is not None, "Test dataset is None"

    trainer_config = CNNParallelTrainerConfig(batch_size=batch_size)
    trainer = CNNParallelTrainer(model, trainer_config)
    trainer.train_and_evaluate(dataset.train, dataset.test)
