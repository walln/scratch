import jax

import optax
from scratch.deep_learning.trainer.trainer_module import TrainState, TrainerModule
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, List
from optax import GradientTransformation


class ImageClassificationBatch:
    inputs: jnp.ndarray
    targets: jnp.ndarray


class ImageClassificationTrainer(TrainerModule):
    num_classes: int

    def __init__(
        self,
        model: nn.Module,
        input_shape: tuple,
        optimizer: Optional[GradientTransformation] = None,
        callbacks: Optional[List] = None,
        num_classes: int = 10,
    ):
        self.num_classes = num_classes
        super(ImageClassificationTrainer, self).__init__(
            model, input_shape, optimizer, callbacks
        )

    def create_training_function(self):
        """Creates and returns a function for the training step."""
        num_classes = self.num_classes

        def apply_model(state: TrainState, inputs: jnp.ndarray, targets: jnp.ndarray):
            def loss_fn(params, input, target):
                logits = state.apply_fn({"params": params}, input)
                one_hot = jax.nn.one_hot(target, num_classes=num_classes)
                loss = jnp.mean(optax.softmax_cross_entropy(logits, one_hot))
                return loss, logits

            # Vectorize the loss function over the batch dimension.
            batched_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0))

            # Compute the mean loss and logits over the batch.
            batch_loss, batch_logits = batched_loss_fn(state.params, inputs, targets)

            # Calculate mean loss and accuracy over the batch
            loss = jnp.mean(batch_loss)
            accuracy = jnp.mean(jnp.argmax(batch_logits, -1).ravel() == targets)

            # Compute gradients
            def loss_for_grad(params):
                batch_losses, _ = batched_loss_fn(params, inputs, targets)
                return jnp.mean(batch_losses), None

            grads = jax.grad(loss_for_grad, has_aux=True)(state.params)[0]

            metrics = {"loss": loss, "accuracy": accuracy}
            return grads, metrics

        @jax.jit
        def train_step(state: TrainState, batch: ImageClassificationBatch):
            inputs, targets = batch
            grads, metrics = apply_model(state, inputs, targets)
            state = state.apply_gradients(grads=grads)

            metrics = {
                "loss": jax.numpy.mean(metrics["loss"]),
                "accuracy": jax.numpy.mean(metrics["accuracy"]),
            }

            return state, metrics

        return train_step
