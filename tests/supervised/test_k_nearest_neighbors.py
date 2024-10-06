"""Tests for k-nearest neighbors."""

import jax.numpy as jnp
import pytest

from scratch.supervised.k_nearest_neighbors import knn
from tests.utils import classification_dataset


@pytest.fixture
def classification_data() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return a classification dataset."""
    X_train, X_test, y_train, y_test = classification_dataset()
    return X_train, X_test, y_train, y_test


@pytest.mark.parametrize("k", [3, 5, 7])
def test_knn(classification_data, k):
    """Test the k-nearest neighbors model."""
    X_train, X_test, y_train, y_test = classification_data

    y_pred = knn(X_train, y_train, X_test, k=k)
    accuracy = jnp.mean(y_pred == y_test)
    assert accuracy > 0.75
