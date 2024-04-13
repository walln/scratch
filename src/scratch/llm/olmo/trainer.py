"""Trainer for olmo model."""

from dataclasses import dataclass

from scratch.llm.olmo.modeling.model import OLMo


@dataclass
class TrainConfig:
    """Configuration for training olmo model."""

    n_epochs: int
    batch_size: int
    lr: float


@dataclass
class Trainer:
    """Trainer for olmo model."""

    config: TrainConfig
    model: OLMo

    def fit(self):
        """Fit the model."""
        pass
