"""Test regression models."""

import jax.numpy as jnp
import pytest
from scratch.supervised.regression import (
    elastic_net_regression,
    lasso_regression,
    linear_regression,
    logistic_regression,
    predict,
    predict_logistic,
    ridge_regression,
)
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

from tests.utils import classification_dataset, regression_dataset


@pytest.fixture()
def regression_data() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return a regression dataset."""
    X_train, X_test, y_train, y_test = regression_dataset()
    return X_train, X_test, y_train, y_test


@pytest.fixture()
def classification_data() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return a classification dataset."""
    X_train, X_test, y_train, y_test = classification_dataset()
    return X_train, X_test, y_train, y_test


def test_linear_regression(regression_data):
    """Test the linear regression model."""
    X_train, X_test, y_train, y_test = regression_data

    weights = linear_regression(X_train, y_train, n_iterations=100)
    predictions = predict(X_test, weights)

    mse = mean_squared_error(y_test, predictions)
    assert mse > 0.00001
    assert mse < 1

    r2 = r2_score(y_test, predictions)
    assert r2 >= 0.9
    assert r2 < 1.0


def test_lasso_regression(regression_data):
    """Test the lasso regression model."""
    X_train, X_test, y_train, y_test = regression_data

    weights = lasso_regression(X_train, y_train, n_iterations=350)
    predictions = predict(X_test, weights)

    mse = mean_squared_error(y_test, predictions)
    assert mse > 0.00001
    assert mse < 1

    r2 = r2_score(y_test, predictions)
    assert r2 >= 0.9
    assert r2 < 1.0


def test_ridge_regression(regression_data):
    """Test the ridge regression model."""
    X_train, X_test, y_train, y_test = regression_data

    weights = ridge_regression(X_train, y_train, n_iterations=350)
    predictions = predict(X_test, weights)

    mse = mean_squared_error(y_test, predictions)
    assert mse > 0.00001
    assert mse < 1

    r2 = r2_score(y_test, predictions)
    assert r2 >= 0.9
    assert r2 < 1.0


def test_elastic_net(regression_data):
    """Test the elastic net model."""
    X_train, X_test, y_train, y_test = regression_data

    weights = elastic_net_regression(X_train, y_train, n_iterations=350)
    predictions = predict(X_test, weights)

    mse = mean_squared_error(y_test, predictions)
    assert mse > 0.00001
    assert mse < 1

    r2 = r2_score(y_test, predictions)
    assert r2 >= 0.9
    assert r2 < 1.0


def test_logistic_regression(classification_data):
    """Test the logistic regression model."""
    X_train, X_test, y_train, y_test = classification_data

    weights = logistic_regression(X_train, y_train, n_iterations=100)
    predictions = predict_logistic(X_test, weights)

    acc = accuracy_score(y_test, predictions)

    assert acc >= 0.9
    assert acc < 1.0

    f1 = f1_score(y_test, predictions)
    assert f1 >= 0.9
    assert f1 < 1.0
