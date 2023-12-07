from typing import Any, Dict, List, Optional, Tuple

import flax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from jax import random
from loguru import logger
from optax import GradientTransformation, adam
from tqdm import tqdm

from scratch.datasets.dataset import Dataloader, Dataset


class TrainState(train_state.TrainState):
    rng: random.PRNGKey  # TODO: Figure out what type this should be
    mutable_variables: Any  # TODO: Figure out what type this should be


class TrainerModule:
    def __init__(
        self,
        model: nn.Module,
        input_shape: tuple,
        optimizer: Optional[GradientTransformation] = None,
        callbacks: Optional[List] = None,
    ):
        self._initialize_logger()
        self._initialize_model(model=model, input_shape=input_shape)
        self._initialize_optimizer(optimizer=optimizer)
        self._initalize_callbacks(callbacks=callbacks)
        self.create_steps()

    def _initialize_logger(self):
        self.logger = logger
        self.logger.remove()
        self.logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
        self.logger.info("Initialized TrainerModule")

    def _initialize_model(self, model: nn.Module, input_shape: Tuple):
        self.logger.info("Initializing model")
        self.model = model
        # TODO: add support for seeding
        model_rng = random.PRNGKey(0xFFFF)
        model_rng, init_rng = random.split(model_rng)
        sample_input = jnp.ones(input_shape)
        variables = self.run_model_init(sample_input, init_rng)

        if isinstance(variables, flax.core.FrozenDict):
            mutable_variables, params = variables.pop("params")
        else:
            params = variables.pop("params")
            mutable_variables = variables

        if len(mutable_variables) == 0:
            mutable_variables = None

        self.logger.info(f"Initialized Model: {self.model.__class__.__name__}")

        tx = optax.adam(learning_rate=1e-3)

        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            mutable_variables=mutable_variables,
            rng=model_rng,
            tx=tx,
        )

    def run_model_init(
        self, sample_input: jnp.ndarray, init_rng: random.KeyArray
    ) -> Dict:
        """The model initialization call.

        Args:
            sample_input: An input to the model with which the shapes are inferred.
            init_rng: A jax.random.PRNGKey.

        Returns:
            The initialized variable dictionary.
        """
        variables = self.model.init({"params": init_rng}, sample_input, train=False)
        if not isinstance(variables, flax.core.frozen_dict.FrozenDict):
            variables = flax.core.frozen_dict.freeze(variables)
        return variables

    def _initialize_optimizer(self, optimizer: GradientTransformation):
        self.logger.info("Initializing optimizer")
        if optimizer is None:
            self.logger.info(
                "No optimizer provided, falling back to default adam with lr=1e-3"
            )

            optimizer = adam(learning_rate=1e-3)
        self.optimizer = optimizer

    def _initalize_callbacks(self, callbacks: list = None):
        self.logger.info("Initializing callbacks")

        if callbacks is None:
            self.logger.info("No callbacks provided")
            self.callbacks = []
        else:
            for callback in callbacks:
                self.logger.info(f"Initializing {callback.__class__.__name__}")
            self.callbacks = callbacks

    def train(self, dataset: Dataset, epochs: int = 1):
        self.logger.info(f"Training for {epochs} epochs")
        epoch_monitor = tqdm(range(epochs), desc="Epochs")
        for epoch in epoch_monitor:
            self.state, _ = self.train_epoch(self.state, dataset.train, epoch)

    def train_epoch(self, state: TrainState, train_loader: Dataloader, epoch: int):
        self.logger.info(f"Training epoch {epoch}")
        batch_monitor = tqdm(train_loader, desc="Batches", leave=False)
        epoch_metrics = []
        for batch in batch_monitor:
            train_step = self.train_step
            state, step_metrics = train_step(state, batch)
            epoch_metrics.append(step_metrics)

            # Loss will always be in the metrics
            # If accuracy is in the metrics, then add it to the progress bar as well
            if "accuracy" in step_metrics:
                accuracy_str = f"Accuracy: {step_metrics['accuracy']:.4f}"
                loss_str = f"Loss: {step_metrics['loss']:.4f}"
                batch_monitor.set_postfix_str(f"{accuracy_str} {loss_str}")
            else:
                batch_monitor.set_postfix_str(f"Loss: {step_metrics['loss']:.4f}")

        # TODO: Compute validation metrics instead of just last step metrics
        metric_logs = [f"{k}: {v:.4f}" for k, v in step_metrics.items()]
        self.logger.info(f"Completed epoch {epoch} - {' '.join(metric_logs)}")

        return state, epoch_metrics

    def create_steps(self):
        train_step = self.create_training_function()
        self.train_step = train_step

    def create_training_function(self):
        raise NotImplementedError()

    def create_validation_function(self):
        raise NotImplementedError()

    def create_test_function(self):
        raise NotImplementedError()
