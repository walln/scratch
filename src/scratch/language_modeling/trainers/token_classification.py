"""Token classification trainer using NNX and SPMD parallelism.

This module provides a trainer for training token classification models using the
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
from scratch.datasets.token_classification_dataset import TokenClassificationBatch
from scratch.trainer import SupervisedTrainer, SupervisedTrainerConfig, TrainState

M = TypeVar("M", bound=nnx.Module)


@dataclass
class TokenClassificationTrainerConfig(SupervisedTrainerConfig):
    """Configuration for the TokenClassificationTrainer."""

    learning_rate: float = 0.001
    num_labels: int = 2


class TokenClassificationTrainer(SupervisedTrainer[M]):
    """Trainer for token classification tasks."""

    trainer_config: TokenClassificationTrainerConfig

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

    def _create_attention_mask(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        """Creates an attention mask from input IDs."""
        mask = (input_ids != 0).astype(jnp.int32)
        return mask[:, None, None, :]  # Expand dimensions to match attention mask shape

    def train(self, train_loader: DataLoader[TokenClassificationBatch]):
        """Trains the model on the entire training dataset."""

        @nnx.jit
        def train_step(model: M, train_state: TrainState, batch: dict):
            def loss_fn(model: Callable):
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    train=True,
                )
                loss = optax.softmax_cross_entropy_with_integer_labels(
                    logits=logits, labels=batch["labels"]
                ).mean()
                return loss, logits

            grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
            (loss, logits), grads = grad_fn(model)

            train_state.update(
                grads=grads,
                loss=loss,
                logits=logits,
                labels=labels,
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
                    input_ids = jax.device_put(
                        batch["input_ids"].numpy(), input_sharding
                    )
                    labels = jax.device_put(batch["labels"].numpy(), target_sharding)
                    attention_mask = self._create_attention_mask(input_ids)

                    train_batch = {
                        "input_ids": input_ids,
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
        test_loader: DataLoader[TokenClassificationBatch],
    ):
        """Evaluates the model on the entire test dataset.

        Args:
            test_loader: The test data loader
        """

        @nnx.jit
        def eval_step(model: M, train_state: TrainState, batch: dict):
            logits = model(  # type: ignore
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                train=False,
            )
            train_state.update_metrics(loss=0.0, logits=logits, labels=batch["labels"])

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
                    input_ids = jax.device_put(
                        batch["input_ids"].numpy(), input_sharding
                    )
                    labels = jax.device_put(batch["labels"].numpy(), target_sharding)
                    attention_mask = self._create_attention_mask(input_ids)

                    eval_batch = {
                        "input_ids": input_ids,
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
