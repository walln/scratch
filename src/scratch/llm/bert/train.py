"""Train BERT model."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import optax

from scratch.llm.bert.model import Bert, BertConfig


def get_model(seed: int = 0):
    """Configure and initialize BERT model."""
    model_key, _ = jax.random.split(jax.random.PRNGKey(seed), num=2)

    model_config = BertConfig(
        dropout=0.0,
        num_heads=12,
        num_blocks=12,
        embedding_size=768,
        vocab_size=50258,
        max_length=512,
    )
    model = Bert(model_config=model_config, key=model_key)

    return model


@dataclasses.dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer configuration."""

    model: Bert
    train_steps: int
    max_lr: float
    b1: float
    b2: float
    eps: float
    weight_decay: float
    clip_grad_norm: float
    gradient_accumulation_steps: int


def get_optimizer(config: OptimizerConfig):
    """Configure and initialize optimizer."""
    # Create learning rate schedule
    steps = config.train_steps // config.gradient_accumulation_steps // 2
    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=config.max_lr, transition_steps=steps
    )
    decay_fn = optax.linear_schedule(
        init_value=config.max_lr, end_value=0.0, transition_steps=steps
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[steps]
    )

    # Create optimizer
    optim = optax.adamw(
        learning_rate=schedule_fn,
        b1=config.b1,
        b2=config.b2,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )

    # Add gradient clipping
    if config.clip_grad_norm > 0.0:
        optim = optax.chain(optim, optax.clip_by_global_norm(config.clip_grad_norm))

    optim = optax.MultiSteps(optim, every_k_schedule=config.gradient_accumulation_steps)
    opt_state = optim.init(eqx.filter(config.model, eqx.is_inexact_array))

    return optim, opt_state


def make_step(policy: jmp.Policy, tx: optax.MultiSteps):
    """Create training step function."""

    @eqx.filter_value_and_grad
    def compute_loss(model, x, y, masks):
        """Compute cross-entropy loss."""
        x, model = policy.cast_to_compute((x, model))
        logits = jax.vmap(model)(x)
        x = policy.cast_to_output(x)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        return jnp.sum(losses * masks) / masks.sum()

    @eqx.filter_jit
    def train_step(model, x, y, masks, opt_state):
        """Perform single training step."""
        loss, grads = compute_loss(model, x, y, masks)
        updates, opt_state = tx.update(
            updates=grads,
            state=opt_state,
            params=eqx.filter(model, eqx.is_inexact_array),
        )
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    return train_step


if __name__ == "__main__":
    model = get_model()

    opt_config = OptimizerConfig(
        model=model,
        max_lr=1e-3,
        b1=0.9,
        b2=0.98,
        eps=1e-12,  # Seems really small
        clip_grad_norm=0.5,
        weight_decay=0.01,  # Multiply by learning rate
        train_steps=100_000,
        gradient_accumulation_steps=32,
    )
    tx, opt_state = get_optimizer(config=opt_config)

    policy = jmp.Policy(
        param_dtype=jnp.float32, compute_dtype=jnp.float16, output_dtype=jnp.float16
    )

    train_step = make_step(policy=policy, tx=tx)
