"""Train ResNet18 v1.5 on MNIST."""
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int, PyTree

from scratch.datasets.dataset import Dataloader, cifar10_dataset
from scratch.deep_learning.resnet.model import ResNet, resnet18


def cross_entropy(
    y: Int[Array, " batch"],  # noqa: F722
    pred_y: Float[Array, "batch 10"],  # noqa: F722
) -> Float[Array, ""]:  # noqa: F722
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


def loss(
    model: ResNet,
    x: Float[Array, "batch 1 28 28"],  # noqa: F722
    y: Int[Array, " batch"],  # noqa: F722
) -> Float[Array, ""]:  # noqa: F722
    """Compute average cross-entropy loss on a batch.

    Input will be of shape (BATCH_SIZE, 1, 28, 28), but our model expects
    (1, 28, 28), so we use jax.vmap to map our model over the leading (batch) axis.

    Args:
    ----
        model: the ResNet model
        x: input batch
        y: target batch

    Returns:
    -------
        Average cross-entropy loss on the batch.
    """
    pred_y = jax.vmap(model)(x)
    return cross_entropy(y, pred_y)


@eqx.filter_jit
def compute_accuracy(
    model: ResNet,
    x: Float[Array, "batch 1 28 28"],  # noqa: F722
    y: Int[Array, " batch"],  # noqa: F722
) -> Float[Array, ""]:  # noqa: F722
    """Compute average accuracy on a batch."""
    pred_y = jax.vmap(model)(x)
    pred_y = jnp.argmax(pred_y, axis=1)
    return jnp.mean(y == pred_y)


def evaluate(model: ResNet, testloader: Dataloader):
    """Evaluate the model on the test dataset.

    Computes both the average loss and the average accuracy.

    Args:
    ----
        model: the ResNet model
        testloader: the test dataset dataloader
    """
    avg_loss = 0
    avg_acc = 0
    for x, y in testloader:
        # Note that all the JAX operations happen inside `loss` and `compute_accuracy`,
        # and both have JIT wrappers, so this is fast.
        avg_loss += loss(model, x, y)
        avg_acc += compute_accuracy(model, x, y)
    return avg_loss / len(testloader), avg_acc / len(testloader)


def train(
    model: ResNet,
    trainloader: Dataloader,
    testloader: Dataloader,
    optim: optax.GradientTransformation,
    steps: int,
    print_every: int,
):
    """Train ResNet18 v1.5."""
    # Filter arrays within the model to get trainable parameters
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(
        model: ResNet,
        opt_state: PyTree,
        x: Float[Array, "batch 1 28 28"],  # noqa: F722
        y: Float[Array, "batch"],  # noqa: F821
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    # Loop over our training dataset as many times as we need.
    def infinite_trainloader():
        while True:
            yield from trainloader

    for step, (x, y) in zip(range(steps), infinite_trainloader()):
        model, opt_state, train_loss = make_step(model, opt_state, x, y)
        if (step % print_every) == 0 or (step == steps - 1):
            test_loss, test_accuracy = evaluate(model, testloader)
            print(
                f"{step=}, train_loss={train_loss.item()}, "
                f"test_loss={test_loss.item()}, test_accuracy={test_accuracy.item()}"
            )
    return model


BATCH_SIZE = 32
LEARNING_RATE = 3e-4
STEPS = 300
PRINT_EVERY = 10
SEED = 5678


if __name__ == "__main__":
    model = resnet18()
    dataset = cifar10_dataset(batch_size=BATCH_SIZE, shuffle=False)
    optim = optax.adamw(LEARNING_RATE)

    # eqx._pretty_print.tree_pprint(model)

    def count_params(model: eqx.Module):
        """Count the number of parameters in the model."""
        num_params = sum(
            x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
        )
        num_millions = num_params / 1_000_000
        print(f"Model # of parameters: {num_millions:.2f}M")

    count_params(model)

    import flax

    flax.linen.summary.tabulate(
        model, jax.random.key(0), compute_flops=True, compute_vjp_flops=True
    )

    model = train(
        model=model,
        trainloader=dataset.train,
        testloader=dataset.test,
        optim=optim,
        steps=STEPS,
        print_every=PRINT_EVERY,
    )
