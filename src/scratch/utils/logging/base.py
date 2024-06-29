"""Base logger for loggers to implement.

This module provides the base logger class for loggers to implement.
It also provides a no-op logger that does nothing for testing purposes.
"""

from abc import ABC, abstractmethod
from typing import Any

from scratch.utils.logging.console import console


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
