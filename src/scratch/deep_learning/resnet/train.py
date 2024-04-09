"""Train ResNet18 v1.5 on MNIST."""

import os
from argparse import ArgumentParser
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from jaxtyping import Array, Float, PyTree
from loguru import logger
from scratch.datasets.dataset import (
    CustomDataLoader,
    CustomImageClassificationBatch,
    tiny_imagenet_dataset,
)
from scratch.deep_learning.resnet.model import ResNet, ResNet18
from scratch.deep_learning.utils import count_params
from tqdm import tqdm

if TYPE_CHECKING:
    from loguru import Logger


def log_epoch_train(epoch: int, losses: Float[Array, ""], logger: "Logger"):
    """Log epoch training results."""
    logger.info(
        f"Epoch: {epoch} training - " f" loss: {losses.mean().item()}",
    )


def log_epoch_val(
    epoch: int,
    losses: Float[Array, ""],
    accuracies: Float[Array, ""],
    logger: "Logger",
):
    """Log epoch validation results."""
    logger.info(
        f"Epoch: {epoch} validation - "
        f" loss: {losses.mean().item()}"
        f" accuracy: {accuracies.mean().item()}"
    )


def cross_entropy(
    y: Float[Array, " batch 200"],
    pred_y: Float[Array, "batch 200"],
) -> Float[Array, ""]:
    """Compute cross entropy loss on a batch of predictions.

    Args:
    ----
        y: target batch
        pred_y: batch of predictions

    Returns:
    -------
        Cross entropy loss.
    """
    pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
    return -jnp.mean(pred_y)


def compute_loss(
    model: ResNet,
    state: eqx.nn.State,
    x: Float[Array, "batch 1 28 28"],
    y: Float[Array, " batch 200"],
):
    """Compute average cross-entropy loss on a batch.

    Input will be of shape (BATCH_SIZE, 1, 28, 28), but our model expects
    (1, 28, 28), so we use jax.vmap to map our model over the leading (batch) axis.

    Args:
    ----
        model: the ResNet model
        state: the batch state
        x: input batch
        y: target batch

    Returns:
    -------
        Average cross-entropy loss on the batch.
    """
    batch_model = jax.vmap(
        model, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
    )
    logits, state = batch_model(x, state)
    loss, _ = compute_metrics(logits=logits, labels=y)
    return loss, (state, loss)


def compute_metrics(
    *, logits: Float[Array, "batch 200"], labels: Float[Array, "batch 200"]
):
    """Compute metrics for a batch of logits and labels."""
    accuracy = jnp.mean(jnp.argmax(logits, axis=1) == jnp.argmax(labels, axis=1))
    losses = optax.softmax_cross_entropy(logits, labels)
    loss = losses.mean()
    return loss, accuracy


def train_epoch(
    model: ResNet,
    state: eqx.nn.State,
    trainloader: CustomDataLoader[CustomImageClassificationBatch],
    epoch: int,
    opt_state: optax.OptState,
    logger: "Logger",
):
    """Train a resnet model for one epoch.

    Args:
    ----
        model: the ResNet model
        state: the batch state
        trainloader: the training dataset
        epoch: the current epoch
        opt_state: the optimizer state
        logger: the logger

    Returns:
    -------
    The model, batch state, and optimizer state.
    """

    @eqx.filter_jit
    def train_step(
        model: PyTree,
        state: eqx.nn.State,
        opt_state: PyTree,
        x: Float[Array, "batch channels height width"],
        y: Float[Array, "batch classes"],
    ) -> tuple[PyTree, eqx.nn.State, PyTree, Float[Array, "batch classes"]]:
        grads, (state, loss) = eqx.filter_grad(compute_loss, has_aux=True)(
            model, state, x, y
        )
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, state, opt_state, loss

    # Set to training mode
    model = eqx.nn.inference_mode(model, value=False)

    train_bar = tqdm(trainloader, leave=False, desc="Training")
    losses = jnp.array([])
    for batch in train_bar:
        x, y = batch.unpack()
        model, state, opt_state, train_loss = train_step(model, state, opt_state, x, y)
        losses = jnp.append(losses, train_loss)
        train_bar.set_postfix_str(f"train_loss={train_loss.item()}")

    log_epoch_train(epoch, losses, logger)

    return model, state, opt_state


def validate_epoch(
    model: ResNet,
    state: eqx.nn.State,
    testloader: CustomDataLoader[CustomImageClassificationBatch],
    epoch: int,
    logger: "Logger",
):
    """Validate a model on a test set.

    Args:
    ----
        model: the ResNet model
        state: the batch state
        testloader: the test dataset
        epoch: the current epoch
        logger: the logger

    Returns:
    -------
    The model and batch state.
    """

    @eqx.filter_jit
    def validate_step(
        model: eqx.Partial,
        xs: Float[Array, "batch channels width height"],
        ys: Float[Array, "batch classes"],
    ) -> tuple[Float[Array, "batch"], Float[Array, "batch"]]:
        logits, _ = jax.vmap(model)(xs)
        loss, accuracy = compute_metrics(logits=logits, labels=ys)
        return loss, accuracy

    # Set to inference mode
    inference_model = eqx.nn.inference_mode(model)
    inference_model = eqx.Partial(inference_model, state=state)
    validation_bar = tqdm(testloader, leave=False, desc="Validation")

    accuracies = jnp.array([])
    losses = jnp.array([])
    for batch in validation_bar:
        x, y = batch.unpack()
        test_loss, accuracy = validate_step(inference_model, x, y)
        accuracies = jnp.append(accuracies, accuracy)
        losses = jnp.append(losses, test_loss)
        validation_bar.set_postfix_str(
            f"test_loss={test_loss.item()}, test_accuracy={accuracy.item()}"
        )

    log_epoch_val(epoch, losses, accuracies, logger)

    return model, state


def train_and_evaluate(
    model: ResNet,
    state: eqx.nn.State,
    trainloader: CustomDataLoader[CustomImageClassificationBatch],
    testloader: CustomDataLoader[CustomImageClassificationBatch],
    optim: optax.GradientTransformation,
    epochs: int,
    checkpointer: ocp.CheckpointManager,
    logger: "Logger",
):
    """Train ResNet18 v1.5."""
    # Filter arrays within the model to get trainable parameters
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    epoch_bar = tqdm(range(epochs), leave=True, desc="ResNet training - Epoch")
    for epoch in epoch_bar:
        epoch_bar.set_postfix_str("Training")
        model, state, opt_state = train_epoch(
            model=model,
            state=state,
            trainloader=trainloader,
            epoch=epoch,
            opt_state=opt_state,
            logger=logger,
        )

        epoch_bar.set_postfix_str("Validating")
        model, state = validate_epoch(
            model=model, state=state, testloader=testloader, epoch=epoch, logger=logger
        )

        checkpointer.save(
            epoch, {"model": model, "state": state, "opt_state": opt_state}, force=True
        )

    return model


def initialize_model_and_opt(
    SEED: int, LEARNING_RATE: float
) -> tuple[ResNet, eqx.nn.State, optax.GradientTransformation]:
    """Initialize model and optimizer."""
    key = jax.random.PRNGKey(SEED)
    key, subkey = jax.random.split(key, 2)
    model, state = eqx.nn.make_with_state(ResNet18)(num_classes=200, key=subkey)
    optim = optax.adamw(LEARNING_RATE)

    return model, state, optim


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=5678)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="scratch/deep_learning/resnet/checkpoints",
    )
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    EPOCHS = args.epochs
    SEED = args.seed
    CHECKPOINT_PATH = os.path.join(os.getcwd(), args.checkpoint_path)

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

    logger.info("Intializing ResNet18 v1.5 training with configuration:")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Learning rate: {LEARNING_RATE}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Seed: {SEED}")
    logger.info(f"Checkpoint path: {CHECKPOINT_PATH}")
    logger.info("-" * 80)

    logger.info("Loading dataset: tiny-imagenet-200")
    dataset = tiny_imagenet_dataset(batch_size=BATCH_SIZE, shuffle=True)

    logger.info("Initializing model and optimizer")
    model, state, optim = initialize_model_and_opt(SEED, LEARNING_RATE)

    param_count, param_count_mils = count_params(model)
    logger.info(f"Model configuration loaded with: {param_count_mils:.2f}M parameters")

    logger.info("-" * 80)

    trainloader = dataset.train
    testloader = dataset.test

    if not testloader:
        raise ValueError("No testloader found.")

    checkpointer_options = ocp.CheckpointManagerOptions(
        save_interval_steps=1, max_to_keep=1
    )
    checkpointer = ocp.CheckpointManager(
        CHECKPOINT_PATH, ocp.PyTreeCheckpointer(), options=checkpointer_options
    )

    model = train_and_evaluate(
        model=model,
        state=state,
        trainloader=trainloader,
        testloader=testloader,
        optim=optim,
        epochs=EPOCHS,
        checkpointer=checkpointer,
        logger=logger,
    )
