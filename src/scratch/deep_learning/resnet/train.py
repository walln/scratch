"""Train ResNet18 v1.5 on MNIST."""

import os
from typing import Annotated

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import typer
from jaxtyping import Array, Float, PyTree
from rich.console import Console
from rich.progress import Progress
from scratch.datasets.dataset import (
    CustomDataLoader,
    CustomImageClassificationBatch,
    cifar10_dataset,
)
from scratch.deep_learning.resnet.model import ResNet, ResNet18
from scratch.deep_learning.utils import count_params

console = Console()

app = typer.Typer(pretty_exceptions_show_locals=False)


def log_epoch_train(epoch: int, losses: Float[Array, ""]):
    """Log epoch training results."""
    console.print(f"Epoch: {epoch} training - loss: {losses.mean().item()}")


def log_epoch_val(
    epoch: int,
    losses: Float[Array, ""],
    accuracies: Float[Array, ""],
):
    """Log epoch validation results."""
    console.print(
        f"Epoch: {epoch} validation - "
        f" loss: {losses.mean().item()}"
        f" accuracy: {accuracies.mean().item()}"
    )


def cross_entropy(
    y: Float[Array, " batch num_classes"],
    pred_y: Float[Array, "batch num_classes"],
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
    x: Float[Array, "batch channels width height"],
    y: Float[Array, " batch num_classes"],
):
    """Compute average cross-entropy loss on a batch.

    Input will be of shape (BATCH_SIZE, channels, width, height), but our model expects
    (1, width, height), so we use jax.vmap to map our model over the leading (batch)
    axis.

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
    *,
    logits: Float[Array, "batch num_classes"],
    labels: Float[Array, "batch num_classes"],
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
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
    progress: Progress,
):
    """Train a resnet model for one epoch.

    Args:
    ----
        model: the ResNet model
        state: the batch state
        trainloader: the training dataset
        epoch: the current epoch
        optim: the optimizer
        opt_state: the optimizer state
        progress: the progress state

    Returns:
    -------
    The model, batch state, and optimizer state.
    """

    @eqx.filter_jit
    def train_step(
        model: PyTree,
        state: eqx.nn.State,
        optim: optax.GradientTransformation,
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

    losses = jnp.array([])
    for batch_idx, batch in progress.track(
        enumerate(trainloader),
        description=f"Training epoch: {epoch}",
        total=len(trainloader),
    ):
        x, y = batch.unpack()
        model, state, opt_state, train_loss = train_step(
            model, state, optim, opt_state, x, y
        )
        losses = jnp.append(losses, train_loss)

        if batch_idx % 250 == 0:
            progress.print(f"train_loss={train_loss.item()}")

    log_epoch_train(epoch, losses)

    return model, state, opt_state


def validate_epoch(
    model: ResNet,
    state: eqx.nn.State,
    testloader: CustomDataLoader[CustomImageClassificationBatch],
    epoch: int,
    progress: Progress,
):
    """Validate a model on a test set.

    Args:
    ----
        model: the ResNet model
        state: the batch state
        testloader: the test dataset
        epoch: the current epoch
        progress: the progress state

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

    accuracies = jnp.array([])
    losses = jnp.array([])
    for batch_idx, batch in progress.track(
        enumerate(testloader),
        description=f"Validation epoch: {epoch}",
        total=len(testloader),
    ):
        x, y = batch.unpack()
        test_loss, accuracy = validate_step(inference_model, x, y)
        accuracies = jnp.append(accuracies, accuracy)
        losses = jnp.append(losses, test_loss)

        if batch_idx % 250 == 0:
            progress.print(
                f"test_loss={test_loss.item()}, test_accuracy={accuracy.item()}"
            )

    log_epoch_val(epoch, losses, accuracies)

    return model, state


def train_and_evaluate(
    model: ResNet,
    state: eqx.nn.State,
    trainloader: CustomDataLoader[CustomImageClassificationBatch],
    testloader: CustomDataLoader[CustomImageClassificationBatch],
    optim: optax.GradientTransformation,
    epochs: int,
    checkpointer: ocp.CheckpointManager,
):
    """Train ResNet18 v1.5."""
    # Filter arrays within the model to get trainable parameters
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    with Progress(console=console, transient=True) as progress:
        epoch_bar = progress.add_task("[cyan]ResNet train and eval", total=epochs)

        for epoch in range(epochs):
            progress.update(epoch_bar, description=f"ResNet training - Epoch {epoch}")
            model, state, opt_state = train_epoch(
                model=model,
                state=state,
                trainloader=trainloader,
                epoch=epoch,
                optim=optim,
                opt_state=opt_state,
                progress=progress,
            )

            progress.update(epoch_bar, description=f"ResNet validating - Epoch {epoch}")
            model, state = validate_epoch(
                model=model,
                state=state,
                testloader=testloader,
                epoch=epoch,
                progress=progress,
            )

            checkpointer.save(
                epoch,
                args=ocp.args.Composite(
                    model=ocp.args.PyTreeSave(model),  # type: ignore orbax broken type
                    state=ocp.args.PyTreeSave(state.tree_flatten()),  # type: ignore orbax broken type
                    opt_state=ocp.args.PyTreeSave(opt_state),  # type: ignore orbax broken type
                ),
            )

            progress.update(epoch_bar, advance=1)

        progress.print("[green]Training complete.")

    return model


def initialize_model_and_opt(
    seed: int, learning_rate: float, num_classes: int
) -> tuple[ResNet, eqx.nn.State, optax.GradientTransformation]:
    """Initialize model and optimizer."""
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key, 2)
    model, state = eqx.nn.make_with_state(ResNet18)(num_classes=num_classes, key=subkey)
    optim = optax.adamw(learning_rate)

    return model, state, optim


@app.command()
def main(
    batch_size: Annotated[int, typer.Option(help="The batch size")] = 64,
    learning_rate: Annotated[float, typer.Option(help="The learning rate")] = 1e-3,
    epochs: Annotated[int, typer.Option(help="The number of epochs")] = 3,
    seed: Annotated[int, typer.Option(help="The random seed")] = 42,
    checkpoint_path: Annotated[
        str, typer.Option(help="The model checkpoint path")
    ] = os.path.join(os.getcwd(), "checkpoints/deep_learning/resnet/"),
):
    """Main training program."""
    console.print("Initializing ResNet18 v1.5 training with configuration:")
    console.print(f"Batch size: {batch_size}")
    console.print(f"Learning rate: {learning_rate}")
    console.print(f"Epochs: {epochs}")
    console.print(f"Seed: {seed}")
    console.print(f"Checkpoint path: {checkpoint_path}")
    console.line()

    console.print("Loading dataset")
    # dataset = tiny_imagenet_dataset(batch_size=batch_size, shuffle=True)
    dataset, dataset_meta = cifar10_dataset(batch_size=batch_size, shuffle=True)
    console.print(f"[green]Dataset loaded: {dataset_meta.name}[/green]")

    console.print("Initializing model and optimizer")
    model, state, optim = initialize_model_and_opt(
        seed, learning_rate, dataset_meta.num_classes
    )

    param_count, param_count_mils = count_params(model)
    console.print(
        f"Model configuration loaded with: {param_count_mils:.2f}M parameters"
    )

    console.line()

    trainloader = dataset.train
    testloader = dataset.test

    if not testloader:
        raise ValueError("No testloader found.")

    checkpointer_options = ocp.CheckpointManagerOptions(
        save_interval_steps=1, max_to_keep=1
    )

    checkpointer = ocp.CheckpointManager(
        checkpoint_path,
        item_names=("model", "state", "opt_state"),
        options=checkpointer_options,
    )

    model = train_and_evaluate(
        model=model,
        state=state,
        trainloader=trainloader,
        testloader=testloader,
        optim=optim,
        epochs=epochs,
        checkpointer=checkpointer,
    )

    checkpointer.close()


if __name__ == "__main__":
    app()
