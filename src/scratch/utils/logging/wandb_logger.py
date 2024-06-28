"""Weights and Biases Logger.

This module provides a logger for Weights & Biases. It includes methods for logging
messages and metrics to the console and Weights & Biases.
"""

from typing import Any, Generic, TypeVar

from scratch.utils.logging.base import BaseLogger

ModelConfig = TypeVar("ModelConfig")
TrainerConfig = TypeVar("TrainerConfig")


class WeightsAndBiasesLogger(BaseLogger, Generic[ModelConfig, TrainerConfig]):
    """A logger for Weights & Biases.

    This class provides methods for logging messages and metrics to the console and
    Weights & Biases. Also logs to the console.
    """

    def __init__(
        self, project: str, model_config: ModelConfig, trainer_config: TrainerConfig
    ):
        """Initializes the logger.

        Args:
            project: The name of the project.
            model_config: The configuration for the model.
            trainer_config: The configuration for the trainer.
        """
        super().__init__()
        self.project = project
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.run = self._init_run()

    def _init_run(self):
        """Initializes a new run."""
        import wandb

        return wandb.init(
            project=self.project,
            config={
                "model": self._dataclass_to_dict(self.model_config),
                "trainer": self._dataclass_to_dict(self.trainer_config),
            },
        )

    def _dataclass_to_dict(self, dataclass):
        """Converts a dataclass to a dictionary.

        Args:
            dataclass: The dataclass to convert.

        Returns:
            The dictionary representation of the dataclass.
        """
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
