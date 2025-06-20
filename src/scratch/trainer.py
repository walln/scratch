"""Abstract trainer class for training models using Flax' nnx api.

This module provides an abstract base class for training models.
It includes support for training state management, checkpointing, and logging.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

import jax
import optax
import orbax.checkpoint as ocp
from flax import nnx
from jax.sharding import Mesh
from rich.console import Console
from rich.progress import Progress

from scratch.utils.logging import (
    BaseLogger,
    console,
    get_progress_widgets,
    no_op_logger,
)
from scratch.utils.timer import capture_time

M = TypeVar("M", bound=nnx.Module)


class TrainState(nnx.Optimizer, Generic[M]):
    """Train state for training models.

    This class manages the training state, including the model, optimizer, and metrics.
    """

    model: M
    tx: optax.GradientTransformation
    metrics: nnx.MultiMetric

    def __init__(self, model, tx, metrics: nnx.MultiMetric):
        """Initializes the train state.

        Args:
            model: The model to train.
            tx: The optimizer transformation.
            metrics: The metrics to track.
        """
        self.metrics = metrics
        super().__init__(model, tx)

    def update(self, *, grads, **updates):
        """Updates the train state in-place.

        Args:
            grads: The gradients to update the model with.
            updates: The updates to apply to the metrics.
        """
        self.metrics.update(**updates)
        super().update(grads)

    def update_metrics(self, **updates):
        """Updates the metrics in-place.

        Args:
            updates: The updates to apply to the metrics.
        """
        self.metrics.update(**updates)

    def reset_metrics(self):
        """Resets the metrics."""
        self.metrics.reset()

    def compute_metrics(self):
        """Realizes the metric values.

        Returns:
            The computed metrics.
        """
        return self.metrics.compute()


@dataclass
class BaseTrainerConfig:
    """Base configuration for the trainer."""

    batch_size: int = 64
    """The global batch size to be sharded across all devices."""
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


class BaseTrainer(Generic[M], ABC):
    """Abstract trainer class for training models.

    This class provides an abstract base for creating trainers for various models.
    It includes methods for creating the training state, logging metrics, and
    checkpointing.
    """

    def __init__(
        self,
        model: M,
        trainer_config: BaseTrainerConfig,
        logger: BaseLogger | None = None,
    ):
        """Initializes a trainer for a specific objective.

        Args:
            model: The model to train.
            trainer_config: The configuration for the trainer.
            logger: The logger to use.
        """
        self.model = model
        self.trainer_config = trainer_config
        self.console = console
        self.checkpoint_manager = self._setup_checkpoint_manager()
        self.train_state = self._create_train_state()
        self.mesh = Mesh(jax.devices(), ("device",))
        self.global_step = 0
        self.logger = logger if logger else no_op_logger

        self.progress = Progress(
            *get_progress_widgets(), console=self.console, transient=True
        )

    @abstractmethod
    def _create_train_state(self) -> TrainState[M]:
        """Creates the initial training state.

        This method should be implemented by subclasses to initialize the training
        state.

        Returns:
            The initial training state.
        """
        pass

    @abstractmethod
    def train(self, train_loader: Any):
        """Train the model for one pass over the training set."""
        pass

    @abstractmethod
    def eval(self, test_loader: Any):
        """Evaluate the model on the test set."""
        pass

    def log_metrics(
        self,
        step_type: Literal["train", "eval"],
        log_to_console=True,
        reset_metrics=True,
    ):
        """Logs the metrics from the training state and resets them.

        Also returns a dict of the computed metrics.

        Args:
            step_type: The type of step
            log_to_console: Whether to log to the console
            reset_metrics: Whether to reset the metrics

        Returns:
            A dict of the computed metrics.
        """
        reported_metrics = {
            f"{step_type}_{metric}": value
            for metric, value in self.train_state.compute_metrics().items()
            if not (step_type == "eval" and metric == "loss")
        }

        self.logger.log_metrics(reported_metrics, self.global_step)

        if log_to_console:
            for metric in reported_metrics:
                self.logger.log(f"{metric}: {reported_metrics[metric]}")

        computed_metrics = self.train_state.compute_metrics()

        if reset_metrics:
            self.train_state.reset_metrics()

        return computed_metrics

    @property
    def checkpointing_enabled(self):
        """Whether checkpointing is enabled.

        Returns:
            A boolean indicating whether checkpointing is enabled.
        """
        return self.trainer_config.save_checkpoint

    def _setup_checkpoint_manager(self):
        """Sets up the checkpoint manager.

        Returns:
            The checkpoint manager.
        """
        opts = ocp.CheckpointManagerOptions(max_to_keep=3, cleanup_tmp_directories=True)
        return ocp.CheckpointManager(
            self.trainer_config.checkpoint_path,
            options=opts,
        )

    def save_checkpoint(self, step: int, metrics: dict):
        """Saves a checkpoint of the model and optimizer state.

        Args:
            step: The current step.
            metrics: The metrics to save.
        """
        self.logger.log(f"Saving checkpoint at step {step}")
        state = nnx.state(self.model)
        opt_state = self.train_state.opt_state

        # Create the checkpoint data structure
        checkpoint_data = {
            "model": state,
            "opt_state": opt_state,
        }

        self.checkpoint_manager.save(
            step=step,
            items=checkpoint_data,
            metrics=metrics,
        )
        self.checkpoint_manager.wait_until_finished()

    def load_checkpoint(self, step: int = 0):
        """Loads the latest checkpoint.

        Args:
            step: The step to load the checkpoint from.
        """
        model = nnx.eval_shape(lambda: self.model)
        state = nnx.state(model)
        opt_state = self.train_state.opt_state

        # Create the target structure for restoration
        target_structure = {
            "model": state,
            "opt_state": opt_state,
        }

        restored = self.checkpoint_manager.restore(
            step=step,
            items=target_structure,
        )

        # Update the model and train state with restored data
        nnx.update(model, restored["model"])
        self.model = model
        self.train_state = self._create_train_state()
        self.train_state.opt_state = restored["opt_state"]
        self.global_step = step

        self.logger.log(f"Loaded checkpoint at step {step}")


@dataclass
class SupervisedTrainerConfig(BaseTrainerConfig):
    """Configuration for the supervised trainer."""

    epochs: int = 1
    """The number of epochs to train for."""


class SupervisedTrainer(BaseTrainer[M], ABC):
    """Abstract trainer class for supervised learning.

    This class extends the BaseTrainer to provide additional functionality
    specific to supervised learning tasks.
    """

    trainer_config: SupervisedTrainerConfig

    def _epoch_size(self, loader):
        """Calculates the size of an epoch.

        Args:
            loader: The data loader.

        Returns:
            The number of samples in an epoch, or None if the loader is empty.
        """
        return len(loader) * self.trainer_config.batch_size if len(loader) else None

    @abstractmethod
    def train(self, train_loader):
        """Train the model for one pass over the training set."""
        pass

    @abstractmethod
    def eval(self, test_loader):
        """Evaluate the model on the test set."""
        pass

    # TODO: resume from checkpoint epoch
    def train_and_evaluate(self, train_loader, test_loader):
        """Trains and evaluates the model for the specified number of epochs.

        Args:
            train_loader: The training data loader
            test_loader: The test data loader
        """
        self.logger.log(f"Running on {jax.default_backend()} backend")
        self.logger.log(f"Using {jax.device_count()} devices")
        self.logger.log("Beginning training and evaluation")

        for epoch in range(self.trainer_config.epochs):
            self.logger.console.rule(f"Epoch {epoch + 1}/{self.trainer_config.epochs}")
            train_loader.set_epoch(epoch)
            test_loader.set_epoch(epoch)
            with self.progress:
                with capture_time() as train_time:
                    self.train(train_loader)
                with capture_time() as eval_time:
                    self.eval(test_loader)
            self.logger.log(f"train_time: {train_time():.2f}s")
            self.logger.log(f"eval_time: {eval_time():.2f}s")
            self.logger.log(f"total_time: {train_time() + eval_time():.2f}s")
