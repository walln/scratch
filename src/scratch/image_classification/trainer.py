"""Image classification trainer using NNX and SPMD parallelism.

This module provides a trainer for training image classification models using the flax
NNX API with SPMD parallelism. The trainer supports distributed training on multiple
devices.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec
from scratch.datasets.dataset import DataLoader
from scratch.datasets.image_classification_dataset import ImageClassificationBatch
from scratch.trainer import (
    SupervisedTrainer,
    SupervisedTrainerConfig,
    TrainState,
)

M = TypeVar("M", bound=nnx.Module)


@dataclass
class ImageClassificationParallelTrainerConfig(SupervisedTrainerConfig):
    """Configuration for the ImageClassificationParallelTrainer."""

    batch_size: int = 64
    """The global batch size to be sharded across all devices."""
    learning_rate: float = 0.005
    """The learning rate for the optimizer."""
    momentum: float = 0.9
    """The momentum for the optimizer."""
    epochs: int = 10
    """The number of epochs to train for."""


class ImageClassificationParallelTrainer(SupervisedTrainer[M]):
    """Trainer for image classification models using NNX and SPMD parallelism.

    This class provides methods for training and evaluating image classification models
    using NNX with support for SPMD parallelism. It includes methods for creating the
    training state, performing training steps, and evaluating the model.
    """

    trainer_config: ImageClassificationParallelTrainerConfig

    def _create_train_state(self):
        """Creates the initial training state.

        This includes the model, optimizer, and metrics.

        Returns:
            The initial training state.
        """
        metrics = nnx.MultiMetric(
            accuracy=nnx.metrics.Accuracy(),
            loss=nnx.metrics.Average("loss"),
        )
        state = TrainState(
            self.model,
            optax.adamw(
                self.trainer_config.learning_rate, self.trainer_config.momentum
            ),
            metrics,
        )
        return state

    def _epoch_size(self, loader):
        """Calculates the size of an epoch.

        Args:
            loader: The data loader.

        Returns:
            The number of samples in an epoch, or None if the loader is empty.
        """
        return len(loader) * self.trainer_config.batch_size if len(loader) else None

    def train(
        self,
        train_loader: DataLoader[ImageClassificationBatch],
    ):
        """Trains the model on the entire training dataset.

        Args:
            train_loader: The training data loader.
        """

        @nnx.jit
        def train_step(
            model: M, train_state: TrainState, inputs: jnp.ndarray, targets: jnp.ndarray
        ):
            def loss_fn(model: Callable):
                logits = model(inputs)
                assert logits.shape == targets.shape
                loss = optax.softmax_cross_entropy(logits=logits, labels=targets).mean()
                return loss, logits

            grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
            (loss, logits), grads = grad_fn(model)

            train_state.update(
                grads=grads,
                loss=loss,
                logits=logits,
                labels=jnp.argmax(targets, axis=-1),
            )

            return loss

        with self.mesh:
            # Define named sharding
            input_sharding = NamedSharding(self.mesh, PartitionSpec("device"))
            target_sharding = NamedSharding(self.mesh, PartitionSpec("device"))

            with self.progress:
                task = self.progress.add_task(
                    "Training",
                    total=self._epoch_size(train_loader),
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
                    train_step(self.model, self.train_state, inputs, targets)
                    self.progress.update(task, advance=self.trainer_config.batch_size)
                    self.global_step += self.trainer_config.batch_size
                    self.log_metrics("train", log_to_console=False, reset_metrics=False)

            self.log_metrics("train")

    def eval(
        self,
        test_loader: DataLoader[ImageClassificationBatch],
    ):
        """Evaluates the model on the entire test dataset.

        Args:
            test_loader: The test data loader
        """

        @nnx.jit
        def eval_step(
            model: M, train_state: TrainState, inputs: jnp.ndarray, targets: jnp.ndarray
        ):
            logits = model(inputs)  # type: ignore - model is a generic and python generic interescion types are not supported
            train_state.update_metrics(
                loss=0.0, logits=logits, labels=jnp.argmax(targets, axis=-1)
            )

        with self.mesh:
            # Define named sharding
            input_sharding = NamedSharding(self.mesh, PartitionSpec("device"))
            target_sharding = NamedSharding(self.mesh, PartitionSpec("device"))

            with self.progress:
                task = self.progress.add_task(
                    "Evaluating",
                    total=self._epoch_size(test_loader),
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
                    eval_step(self.model, self.train_state, inputs, targets)
                    self.progress.update(task, advance=self.trainer_config.batch_size)

            metrics = self.log_metrics("eval")
            if self.checkpointing_enabled:
                self.save_checkpoint(
                    step=self.global_step,
                    metrics=metrics,
                )
