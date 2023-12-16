"""Train a CNN on MNIST using Flax."""
from dataclasses import dataclass
from functools import partial
from typing import Dict

import flax
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax.training.early_stopping import EarlyStopping
from jax import jit
from jaxtyping import Array
from tqdm import tqdm

from scratch.datasets.dataset import mnist_dataset
from scratch.deep_learning.cnn.model import CNN


class TrainState(train_state.TrainState):
    """Custom TrainState class for Flax."""

    ...


def create_train_state(rng: Array, learning_rate: float, momentum: float):
    """Create initial `TrainState`."""
    model = CNN(num_classes=10)
    variables = model.init(rng, jnp.ones([1, 28, 28, 1]))
    params = variables["params"]
    tx = optax.sgd(learning_rate, momentum)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@dataclass
class TrainEvaluateConfig:
    """Configuration for training and evaluation."""

    num_epochs: int
    batch_size: int
    learning_rate: float
    momentum: float
    seed: int = 0xFFFF


@jit
def train_step(state: TrainState, batch: Dict[str, jnp.ndarray]):
    """Train for a single step.

    Args:
    ----
        state: the current training state
    batch: the batch of data to train on
    """

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["image"])
        one_hot_labels = jax.nn.one_hot(batch["label"], logits.shape[1])
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")

    loss, logits = aux
    new_state = state.apply_gradients(grads=grads)

    metrics = {}
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == batch["label"])
    metrics["accuracy"] = accuracy
    metrics["loss"] = loss
    return new_state, metrics


@jit
def eval_step(state: TrainState, batch: Dict[str, jnp.ndarray]):
    """Evaluate for a single step.

    Args:
    ----
        state: the current training state
    batch: the batch of data to evaluate on
    """
    variables = {"params": state.params}
    logits = state.apply_fn(variables, batch["image"], train=False, mutable=False)
    one_hot_labels = jax.nn.one_hot(batch["label"], logits.shape[1])
    loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels).mean()
    metrics = {}
    metrics["accuracy"] = jnp.mean(jnp.argmax(logits, axis=-1) == batch["label"])
    metrics["loss"] = loss

    return metrics


def train_and_evaluate(config: TrainEvaluateConfig):
    """Train and evaluate a CNN on MNIST.

    Args:
    ----
        config: the configuration for training and evaluation
    """
    num_devices = jax.device_count()
    dataset = mnist_dataset(batch_size=config.batch_size, shuffle=True)
    train_dataloader = dataset.train
    test_dataloader = dataset.test

    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)

    state = create_train_state(init_rng, config.learning_rate, config.momentum)

    state = flax.jax_utils.replicate(state)

    parallel_train_step = jax.pmap(partial(train_step), axis_name="batch")
    parallel_eval_step = jax.pmap(eval_step, axis_name="batch")

    early_stop = EarlyStopping(min_delta=1e-3, patience=2)

    for epoch in range(1, config.num_epochs + 1):
        # -----------------------------------------------------#
        # Training                                             #
        # -----------------------------------------------------#
        train_loader = tqdm(
            train_dataloader, desc=f"Training Epoch {epoch}", leave=False
        )
        train_accuracy = jnp.array([])
        train_loss = jnp.array([])
        for batch in train_loader:
            image, label = batch.unpack()
            if image.shape[0] % num_devices != 0:
                # Batch size must be divisible by the number of devices
                continue

            # Reshape images from (num_devices * batch_size, height, width, channels)
            # to (num_devices, batch_size, height, width, channels).
            # The first dimension will be mapped across devices.
            image = jnp.reshape(image, (num_devices, -1) + image.shape[1:])
            label = jnp.reshape(label, (num_devices, -1) + label.shape[1:])

            state, metrics = parallel_train_step(
                state, {"image": image, "label": label}
            )

            train_accuracy = jnp.append(train_accuracy, metrics["accuracy"])
            train_loss = jnp.append(train_loss, metrics["loss"])

            train_loader.set_postfix_str(
                f"Accuracy: {metrics['accuracy'][0]:.4f} Loss: {metrics['loss'][0]:.4f}"
            )

        train_average_accuracy = jnp.mean(train_accuracy)
        print(f"Epoch: {epoch}, Training Accuracy: {train_average_accuracy:.4f}")

        # -----------------------------------------------------#
        # Validation                                           #
        # -----------------------------------------------------#
        validation_loader = tqdm(
            test_dataloader, desc=f"Validation Epoch {epoch}", leave=False
        )
        validation_accuracy = jnp.array([])
        validation_loss = jnp.array([])
        for batch in validation_loader:
            image, label = batch.unpack()
            if image.shape[0] % num_devices != 0:
                # Batch size must be divisible by the number of devices
                continue

            # Reshape images from (num_devices * batch_size, height, width, channels)
            # to (num_devices, batch_size, height, width, channels).
            # The first dimension will be mapped across devices.
            image = jnp.reshape(image, (num_devices, -1) + image.shape[1:])
            label = jnp.reshape(label, (num_devices, -1) + label.shape[1:])

            metrics = parallel_eval_step(state, {"image": image, "label": label})

            validation_accuracy = jnp.append(validation_accuracy, metrics["accuracy"])
            validation_loss = jnp.append(validation_loss, metrics["loss"])

            validation_loader.set_postfix_str(
                f"Accuracy: {metrics['accuracy'][0]:.4f} Loss: {metrics['loss'][0]:.4f}"
            )

        validation_average_accuracy = jnp.mean(validation_accuracy)
        print(f"Epoch: {epoch}, Validation Accuracy: {validation_average_accuracy:.4f}")

        early_stop = early_stop.update(validation_average_accuracy)
        if early_stop.should_stop:
            print(f"Met early stopping criteria, breaking at epoch {epoch}")
            break


if __name__ == "__main__":
    print(jax.devices())
    train_and_evaluate(
        TrainEvaluateConfig(
            num_epochs=10, batch_size=256, learning_rate=0.01, momentum=0.9
        )
    )
