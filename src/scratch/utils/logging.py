"""Logging utilities for the project."""

import logging

from rich.console import Console
from rich.progress import MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn


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
