"""Question answering trainer using NNX and SPMD parallelism.

This module provides a trainer for training question answering models using the
flax NNX API with SPMD parallelism. The trainer supports distributed training on
multiple devices.
"""

from dataclasses import dataclass
from typing import TypeVar

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec

from scratch.datasets.dataset import DataLoader
from scratch.datasets.question_answering_dataset import QuestionAnsweringBatch
from scratch.trainer import SupervisedTrainer, SupervisedTrainerConfig, TrainState

M = TypeVar("M", bound=nnx.Module)


@dataclass
class QuestionAnsweringTrainerConfig(SupervisedTrainerConfig):
    """Configuration for the QuestionAnsweringTrainer."""

    learning_rate: float = 0.001


class QuestionAnsweringTrainer(SupervisedTrainer[M]):
    """Trainer for question answering tasks."""

    trainer_config: QuestionAnsweringTrainerConfig

    def _create_train_state(self):
        """Creates the initial training state."""
        metrics = nnx.MultiMetric(
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

    def train(self, train_loader: DataLoader[QuestionAnsweringBatch]):
        """Trains the model on the entire training dataset."""

        @nnx.jit
        def train_step(model: M, train_state: TrainState, batch: dict):
            def loss_fn(model: nnx.Module):
                start_logits, end_logits = model(  # type: ignore
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    train=True,
                )
                start_loss = optax.softmax_cross_entropy_with_integer_labels(
                    logits=start_logits, labels=batch["start_positions"]
                ).mean()
                end_loss = optax.softmax_cross_entropy_with_integer_labels(
                    logits=end_logits, labels=batch["end_positions"]
                ).mean()
                loss = (start_loss + end_loss) / 2
                return loss, (start_logits, end_logits)

            grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
            (loss, (start_logits, end_logits)), grads = grad_fn(model)

            train_state.update(
                grads=grads,
                loss=loss,
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
                    start_positions = jax.device_put(
                        batch["start_positions"].numpy(), target_sharding
                    )
                    end_positions = jax.device_put(
                        batch["end_positions"].numpy(), target_sharding
                    )
                    attention_mask = self._create_attention_mask(input_ids)

                    train_batch = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "start_positions": start_positions,
                        "end_positions": end_positions,
                    }

                    train_step(self.model, self.train_state, train_batch)
                    self.progress.update(task, advance=self.trainer_config.batch_size)
                    self.global_step += self.trainer_config.batch_size
                    self.log_metrics("train", log_to_console=False, reset_metrics=False)

            self.log_metrics("train")

    def eval(
        self,
        test_loader: DataLoader[QuestionAnsweringBatch],
    ):
        """Evaluates the model on the entire test dataset.

        Args:
            test_loader: The test data loader
        """

        @nnx.jit
        def eval_step(model: M, train_state: TrainState, batch: dict):
            start_logits, end_logits = model(  # type: ignore
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                train=False,
            )
            train_state.update_metrics(
                loss=0.0,
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
                    input_ids = jax.device_put(
                        batch["input_ids"].numpy(), input_sharding
                    )
                    start_positions = jax.device_put(
                        batch["start_positions"].numpy(), target_sharding
                    )
                    end_positions = jax.device_put(
                        batch["end_positions"].numpy(), target_sharding
                    )
                    attention_mask = self._create_attention_mask(input_ids)

                    eval_batch = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "start_positions": start_positions,
                        "end_positions": end_positions,
                    }

                    eval_step(self.model, self.train_state, eval_batch)
                    self.progress.update(task, advance=self.trainer_config.batch_size)

            metrics = self.log_metrics("eval")
            if self.checkpointing_enabled:
                self.save_checkpoint(
                    step=self.global_step,
                    metrics=metrics,
                )
