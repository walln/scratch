"""Logging utilities for the project."""
import logging


def setup_logger():
    """Return a logger."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger
