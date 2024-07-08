"""Tests for support vector machine."""

import jax.numpy as jnp
import pytest
from scratch.supervised.support_vector_machine import (
    concat_weights_and_bias,
    linear_kernel,
    predict_classification,
    predict_regression,
    split_weights_and_bias,
    support_vector_machine_classifier,
    support_vector_machine_regressor,
    svm_classification_loss,
    svm_regression_loss,
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


def test_svm_classification_loss(classification_data):
    """Test the SVM classification loss function."""
    X_train, _, y_train, _ = classification_data
    W = jnp.zeros(X_train.shape[0])
    b = jnp.ones(1)
    params = concat_weights_and_bias(W, b)
    loss = svm_classification_loss(params, linear_kernel, X_train, y_train, C=1.0)
    assert loss > 0


def test_svm_regression_loss(regression_data):
    """Test the SVM regression loss function."""
    X_train, _, y_train, _ = regression_data
    W = jnp.zeros(X_train.shape[0])
    b = jnp.ones(1)
    params = concat_weights_and_bias(W, b)
    loss = svm_regression_loss(params, linear_kernel, X_train, y_train, C=1.0)
    assert loss > 0


def test_train_svm_classifier(classification_data):
    """Test the classifier training function."""
    _, X_test, _, y_test = classification_data
    params = support_vector_machine_classifier(
        X_test, y_test, kernel=linear_kernel, C=1.0, learning_rate=1e-3, num_epochs=10
    )
    assert isinstance(params, jnp.ndarray)
    W, b = split_weights_and_bias(params)
    assert W.shape == (X_test.shape[0],)


def test_train_svm_regressor(regression_data):
    """Test the SVM regressor training function."""
    X_train, _, y_train, _ = regression_data
    params = support_vector_machine_regressor(
        X_train,
        y_train,
        kernel=linear_kernel,
        C=1.0,
        epsilon=0.1,
        learning_rate=1e-3,
        num_epochs=10,
    )
    assert isinstance(params, jnp.ndarray)
    W, b = split_weights_and_bias(params)
    assert W.shape == (X_train.shape[0],)


def test_predict_classification(classification_data):
    """Test the predict function."""
    _, X_test, _, y_test = classification_data
    params = support_vector_machine_classifier(
        X_test, y_test, kernel=linear_kernel, C=1.0, learning_rate=1e-3, num_epochs=10
    )
    assert isinstance(params, jnp.ndarray)
    predictions = predict_classification(params, linear_kernel, X_test)
    assert predictions.shape == y_test.shape
    assert jnp.all(jnp.isin(predictions, jnp.array([-1, 1])))


def test_predict_regression(regression_data):
    """Test the SVM regression predict function."""
    X_train, _, y_train, _ = regression_data
    params = support_vector_machine_regressor(
        X_train,
        y_train,
        kernel=linear_kernel,
        C=1.0,
        epsilon=0.1,
        learning_rate=1e-3,
        num_epochs=10,
    )
    assert isinstance(params, jnp.ndarray)
    predictions = predict_regression(params, linear_kernel, X_train)
    assert predictions.shape == y_train.shape
    assert jnp.all(jnp.isfinite(predictions))
