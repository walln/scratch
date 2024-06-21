"""Image classification trainer."""

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

import jax
import jax.numpy as jnp
import optax
import orbax
import orbax.checkpoint
import orbax.checkpoint as ocp
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from rich.console import Console
from rich.progress import Progress
from scratch.datasets.dataset import DataLoader
from scratch.datasets.image_classification_dataset import ImageClassificationBatch
from scratch.utils.logging import console, get_progress_widgets
from scratch.utils.timer import capture_time

M = TypeVar("M", bound=nnx.Module)


class TrainState(Generic[M], nnx.Optimizer):
    """Train state for the CNN model."""

    model: M
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
class ImageClassificationParallelTrainerConfig:
    """Configuration for the CNNParallelTrainer."""

    batch_size: int = 64
    """The global batch size to be sharded across all devices."""
    learning_rate: float = 0.005
    """The learning rate for the optimizer."""
    momentum: float = 0.9
    """The momentum for the optimizer."""
    epochs: int = 10
    """The number of epochs to train for."""
    console: Console = console
    """The console to use for logging."""
    checkpoint_path: str = "checkpoints/nnx-cnn/mnist"
    """The path to save the checkpoints."""
    save_checkpoint: bool = False
    """Whether to save checkpoints."""
    resume_checkpoint: bool = False
    """Whether to resume training from a checkpoint."""

    def __post_init__(self):
        """Ensure checkpoint_path is absolute."""
        if not os.path.isabs(self.checkpoint_path):
            self.checkpoint_path = os.path.abspath(self.checkpoint_path)


class ImageClassificationParallelTrainer(Generic[M]):
    """Trains a CNN model using NNX and SPMD parallelism."""

    def __init__(
        self, model: M, trainer_config: ImageClassificationParallelTrainerConfig
    ):
        """Initializes the CNN trainer.

        Args:
            model: The CNN model to train
            trainer_config: The configuration for the trainer
        """
        self.console = console
        self.model = model
        self.train_config = trainer_config
        self.jit_train_step = nnx.jit(ImageClassificationParallelTrainer.train_step)
        self.jit_eval_step = nnx.jit(ImageClassificationParallelTrainer.eval_step)
        self.state = self._create_train_state(
            learning_rate=trainer_config.learning_rate, momentum=trainer_config.momentum
        )
        self.mesh = Mesh(jax.devices(), ("x",))

        checkpointer_options = orbax.checkpoint.CheckpointManagerOptions(
            max_to_keep=3, cleanup_tmp_directories=True
        )
        self.checkpoint_manager = ocp.CheckpointManager(
            self.train_config.checkpoint_path,
            item_names=("model", "opt_state", "metadata"),
            options=checkpointer_options,
        )

    def save_checkpoint(self, step: int, metrics: dict):
        """Saves a checkpoint of the model and optimizer state.

        Args:
            step: The current step
            metrics: The metrics to save
        """
        self.console.log(f"Saving checkpoint at step {step}")
        state = nnx.state(self.model)
        opt_state = self.state.opt_state
        metadata = {
            "batch_size": self.train_config.batch_size,
            "momentum": self.train_config.momentum,
            "learning_rate": self.train_config.learning_rate,
        }
        self.checkpoint_manager.save(
            step=step,
            metrics=metrics,
            args=orbax.checkpoint.args.Composite(
                model=orbax.checkpoint.args.PyTreeSave(state),  # type: ignore - orbax types kinda suck
                opt_state=orbax.checkpoint.args.PyTreeSave(opt_state),  # type: ignore - orbax types kinda suck
                metadata=orbax.checkpoint.args.JsonSave(metadata),  # type: ignore - orbax types kinda suck
            ),
        )
        self.checkpoint_manager.wait_until_finished()

    def load_checkpoint(self, step: int = 0):
        """Loads the latest checkpoint."""
        model = nnx.eval_shape(lambda: self.model)
        state = nnx.state(model)

        opt_state = self.state.opt_state

        metadata = {
            "batch_size": self.train_config.batch_size,
            "momentum": self.train_config.momentum,
            "learning_rate": self.train_config.learning_rate,
        }

        self.checkpoint_manager.restore(
            step=step,
            args=ocp.args.Composite(
                model=ocp.args.PyTreeRestore(state),  # type: ignore - orbax types kinda suck
                opt_state=ocp.args.PyTreeRestore(opt_state),  # type: ignore - orbax types kinda suck
                metadata=ocp.args.JsonRestore(metadata),  # type: ignore - orbax types kinda suck
            ),
        )

        console.log(f"Loaded checkpoint at step {step}")
        console.log(f"Loaded metadata: {metadata}")

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
                    self.train(train_loader, progress=progress, epoch=epoch)
                with capture_time() as eval_time:
                    self.eval(test_loader, progress=progress, epoch=epoch)
            console.log(f"train_time: {train_time():.2f}s")
            console.log(f"eval_time: {eval_time():.2f}s")
            console.log(f"total_time: {train_time() + eval_time():.2f}s")

    def log_metrics(self, step_type: Literal["train", "eval"], progress: Progress):
        """Logs the metrics from the training state and resets them.

        Also returns a dict of the computed metrics.

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
        computed_metrics = self.state.compute_metrics()
        self.state.reset_metrics()
        return computed_metrics

    @staticmethod
    def train_step(
        model: M,
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

        def loss_fn(model: Callable):
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
        epoch: int | None = None,
    ):
        """Trains the model on the entire training dataset.

        Args:
            train_loader: The training data loader
            progress: The progress manager
            epoch: The current epoch
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
        model: M,
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
        logits = model(inputs)  # type: ignore - model is a generic and python generic interescion types are not supported
        train_state.update_metrics(
            loss=0.0, logits=logits, labels=jnp.argmax(targets, axis=-1)
        )

    def eval(
        self,
        test_loader: DataLoader[ImageClassificationBatch],
        *,
        progress: Progress | None = None,
        epoch: int | None = None,
    ):
        """Evaluates the model on the entire test dataset.

        Args:
            test_loader: The test data loader
            progress: The progress manager
            epoch: The current epoch
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
            metrics = self.log_metrics("eval", progress)
            if self.train_config.save_checkpoint:
                self.save_checkpoint(
                    step=epoch * len(test_loader) * self.train_config.batch_size
                    if epoch and len(test_loader)
                    else 0,
                    metrics=metrics,
                )
