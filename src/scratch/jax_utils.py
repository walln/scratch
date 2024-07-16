"""Common JAX utilities."""

from collections.abc import Callable
from typing import Any

import jax


def filter_pytree_by_name(pytree: Any, filter_fn: Callable[[str], bool]) -> Any:
    """Recursively filter a PyTree by node name.

    Args:
        pytree: The input PyTree.
        filter_fn: A function that takes a node name and returns True if the node should
          be kept, False otherwise.

    Returns:
        The filtered PyTree.
    """

    def traverse_fn(path, node):
        name = path[-1] if path else ""
        if filter_fn(name):
            return node
        else:
            return None

    return jax.tree_util.tree_map_with_path(traverse_fn, pytree)
