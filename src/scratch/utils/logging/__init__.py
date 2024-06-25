"""Common logging utilities."""

from scratch.utils.logging.base import BaseLogger as BaseLogger
from scratch.utils.logging.base import NoOpLogger
from scratch.utils.logging.console import console as console
from scratch.utils.logging.progress_bar import (
    get_progress_widgets as get_progress_widgets,
)
from scratch.utils.logging.wandb_logger import (
    WeightsAndBiasesLogger as WeightsAndBiasesLogger,
)

no_op_logger = NoOpLogger()
