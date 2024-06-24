"""Logging utilities for the project."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from rich.console import Console
from rich.progress import MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn

import wandb


def setup_logger():
    """Return a logger."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger


def get_progress_widgets():
    """Return a list of progress bar widgets."""
    return [
        SpinnerColumn(),
        *Progress.get_default_columns(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ]


console = Console()


class BaseLogger(ABC):
    """Base logger class."""

    def __init__(self):
        """Initializes the logger."""
        self.console = console

    @abstractmethod
    def log(*objects: Any):
        """Log any objects.

        Args:
            objects: Any objects to log
        """
        pass

    @abstractmethod
    def log_metrics(self, metrics: dict, step: int):
        """Logs the metrics.

        Args:
            metrics: The metrics to log
            step: The step number
        """
        pass


class NoOpLogger(BaseLogger):
    """A logger that does nothing."""

    def log(self, *objects: Any):
        """Logs the messages to the console.

        Args:
            objects: The objects to log
        """
        self.console.log(*objects)

    def log_metrics(self, metrics: dict, step: int):
        """Logs the metrics to Weights & Biases.

        Args:
            metrics: The metrics to log
            step: The step number
        """
        pass


no_op_logger = NoOpLogger()


ModelConfig = TypeVar("ModelConfig")
TrainerConfig = TypeVar("TrainerConfig")


class WeightsAndBiasesLogger(BaseLogger, Generic[ModelConfig, TrainerConfig]):
    """A logger for Weights & Biases."""

    def __init__(
        self, project: str, model_config: ModelConfig, trainer_config: TrainerConfig
    ):
        """Initializes the logger."""
        super().__init__()
        self.project = project
        self.wandb = wandb
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.run = self._init_run()

    def _init_run(self):
        """Initializes a new run."""
        return self.wandb.init(
            project=self.project,
            config={
                "model": self._dataclass_to_dict(self.model_config),
                "trainer": self._dataclass_to_dict(self.trainer_config),
            },
        )

    def _dataclass_to_dict(self, dataclass):
        """Converts a dataclass to a dictionary."""
        return dataclass.__dict__

    def log(self, *objects: Any):
        """Logs the messages to the console.

        Args:
            objects: The objects to log
        """
        self.console.log(*objects)

    def log_metrics(self, metrics: dict, step: int):
        """Logs the metrics to Weights & Biases.

        Args:
            metrics: The metrics to log
            step: The step number
        """
        self.run.log(metrics, step=step)
