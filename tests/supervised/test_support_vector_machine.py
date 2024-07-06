"""Tests for support vector machine."""

import jax.numpy as jnp
import pytest
from scratch.supervised.support_vector_machine import (
    concat_weights_and_bias,
    linear_kernel,
    predict,
    split_weights_and_bias,
    support_vector_machine,
    svm_loss,
)

from tests.utils import classification_dataset, regression_dataset


@pytest.fixture()
def regression_data():
    """Return a regression dataset."""
    X_train, X_test, y_train, y_test = regression_dataset()
    return X_train, X_test, y_train, y_test


@pytest.fixture()
def classification_data() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return a classification dataset."""
    X_train, X_test, y_train, y_test = classification_dataset()
    return X_train, X_test, y_train, y_test


def test_linear_kernel():
    """Test the linear kernel function."""
    x1 = jnp.array([1, 2])
    x2 = jnp.array([3, 4])
    assert linear_kernel(x1, x2) == jnp.dot(x1, x2)


def test_concat_split_params():
    """Test concatenation and splitting of parameters."""
    W = jnp.array([1, 2, 3])
    b = jnp.array([4.0])
    params = concat_weights_and_bias(W, b)
    W_split, b_split = split_weights_and_bias(params)
    assert jnp.array_equal(W, W_split)
    assert b == b_split


def test_svm_loss(classification_data):
    """Test the SVM loss function."""
    X_train, X_test, _, y_test = classification_data
    W = jnp.zeros(X_train.shape[1])
    b = jnp.ones(1)
    params = concat_weights_and_bias(W, b)
    loss = svm_loss(params, linear_kernel, X_test, y_test, C=1.0)
    assert loss > 0


def test_train_svm(classification_data):
    """Test the training function."""
    _, X_test, _, y_test = classification_data
    params = support_vector_machine(
        X_test, y_test, kernel=linear_kernel, C=1.0, learning_rate=1e-3, num_epochs=10
    )
    assert isinstance(params, jnp.ndarray)
    W, b = split_weights_and_bias(params)
    assert W.shape == (X_test.shape[1],)


def test_predict(classification_data):
    """Test the predict function."""
    _, X_test, _, y_test = classification_data
    params = support_vector_machine(
        X_test, y_test, kernel=linear_kernel, C=1.0, learning_rate=1e-3, num_epochs=100
    )
    assert isinstance(params, jnp.ndarray)
    predictions = predict(params, linear_kernel, X_test)
    assert predictions.shape == y_test.shape
    assert jnp.all(jnp.isin(predictions, jnp.array([-1, 1])))
