"""Sequence classification trainer using NNX and SPMD parallelism.

This module provides a trainer for training sequence classification models using the
flax NNX API with SPMD parallelism. The trainer supports distributed training on
multiple devices.
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
from scratch.datasets.sequence_classification_dataset import SequenceClassificationBatch
from scratch.trainer import SupervisedTrainer, SupervisedTrainerConfig, TrainState

M = TypeVar("M", bound=nnx.Module)


@dataclass
class SequenceClassificationTrainerConfig(SupervisedTrainerConfig):
    """Configuration for the SequenceClassificationTrainer."""

    learning_rate: float = 0.001
    num_labels: int = 2


class SequenceClassificationTrainer(SupervisedTrainer[M]):
    """Trainer for sequence classification tasks using BERT."""

    trainer_config: SequenceClassificationTrainerConfig

    def _create_train_state(self):
        """Creates the initial training state."""
        metrics = nnx.MultiMetric(
            accuracy=nnx.metrics.Accuracy(),
            loss=nnx.metrics.Average("loss"),
        )
        state = TrainState(
            self.model,
            optax.adam(self.trainer_config.learning_rate),
            metrics,
        )
        return state

    def train(self, train_loader: DataLoader[SequenceClassificationBatch]):
        """Trains the model on the entire training dataset."""

        @nnx.jit
        def train_step(model: M, train_state: TrainState, batch: dict):
            def loss_fn(model: Callable):
                logits = model(batch, train=True)
                loss = optax.softmax_cross_entropy(
                    logits=logits, labels=batch["labels"]
                ).mean()
                return loss, logits

            grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
            (loss, logits), grads = grad_fn(model)

            train_state.update(
                grads=grads, loss=loss, logits=logits, labels=batch["labels"]
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
                    input_ids = jax.device_put(batch["input_ids"], input_sharding)
                    token_type_ids = jax.device_put(
                        batch["token_type_ids"], input_sharding
                    )
                    attention_mask = jax.device_put(
                        batch["attention_mask"], input_sharding
                    )
                    labels = jax.device_put(batch["label"], target_sharding)

                    train_batch = {
                        "input_ids": input_ids,
                        "token_type_ids": token_type_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                    }

                    train_step(self.model, self.train_state, train_batch)
                    self.progress.update(task, advance=self.trainer_config.batch_size)
                    self.global_step += self.trainer_config.batch_size
                    self.log_metrics("train", log_to_console=False, reset_metrics=False)

            self.log_metrics("train")

    def eval(
        self,
        test_loader: DataLoader[SequenceClassificationBatch],
    ):
        """Evaluates the model on the entire test dataset.

        Args:
            test_loader: The test data loader
        """

        @nnx.jit
        def eval_step(model: M, train_state: TrainState, batch: dict):
            logits = model(batch)  # type: ignore - model is a generic and python generic interescion types are not supported
            train_state.update_metrics(
                loss=0.0, logits=logits, labels=jnp.argmax(batch["labels"], axis=-1)
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
                    input_ids = jax.device_put(batch["input_ids"], input_sharding)
                    token_type_ids = jax.device_put(
                        batch["token_type_ids"], input_sharding
                    )
                    attention_mask = jax.device_put(
                        batch["attention_mask"], input_sharding
                    )
                    labels = jax.device_put(batch["label"], target_sharding)

                    eval_batch = {
                        "input_ids": input_ids,
                        "token_type_ids": token_type_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                    }

                    eval_step(self.model, self.train_state, eval_batch)
                    self.progress.update(task, advance=self.trainer_config.batch_size)

            metrics = self.log_metrics("eval")
            if self.checkpointing_enabled:
                self.save_checkpoint(
                    step=self.global_step,
                    metrics=metrics,
                )
