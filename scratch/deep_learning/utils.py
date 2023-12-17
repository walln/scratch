"""Utility functions for deep learning."""

import equinox as eqx
import jax


def count_params(model: eqx.Module):
    """Count the number of parameters in the model."""
    num_params = sum(
        x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    )
    num_millions = num_params / 1_000_000

    return num_params, num_millions
