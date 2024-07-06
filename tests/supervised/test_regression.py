"""Test regression models."""

from scratch.supervised.regression import (
    elastic_net_regression,
    lasso_regression,
    linear_regression,
    logistic_regression,
    predict,
    predict_logistic,
    ridge_regression,
)
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def regression_dataset(
    n_samples: int = 2000,
    split_size: float = 0.4,
    n_features: int = 1,
    noise: int = 20,
    seed: int = 1,
    shuffle=False,
):
    """Return a regression dataset.

    Args:
        n_samples: Number of samples.
        split_size: Test split size.
        n_features: Number of features.
        noise: Noise level.
        seed: Random seed.
        shuffle: Shuffle the data.

    Returns:
        X_train: Training data.
        X_test: Test data.
        y_train: Training target.
        y_test: Test target.
    """
    X, y = make_regression(  # type: ignore broken types
        n_samples=n_samples, n_features=n_features, noise=noise, random_state=seed
    )

    data_scaler = StandardScaler()
    target_scaler = MinMaxScaler()

    X = data_scaler.fit_transform(X)
    y = target_scaler.fit_transform(y.reshape(-1, 1))
    y = y.reshape(-1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_size, shuffle=shuffle
    )

    return X_train, X_test, y_train, y_test


def classification_dataset(
    n_samples: int = 2000,
    split_size: float = 0.4,
    n_features: int = 20,
    seed: int = 1,
    shuffle=False,
):
    """Return a classification dataset.

    Args:
        n_samples: Number of samples.
        split_size: Test split size.
        n_features: Number of features.
        seed: Random seed.
        shuffle: Shuffle the data.

    Returns:
        X_train: Training data.
        X_test: Test data.
        y_train: Training target.
        y_test: Test target.
    """
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, random_state=seed
    )

    data_scaler = StandardScaler()

    X = data_scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_size, shuffle=shuffle
    )

    return X_train, X_test, y_train, y_test


def test_linear_regression():
    """Test the linear regression model."""
    X_train, X_test, y_train, y_test = regression_dataset()

    weights = linear_regression(X_train, y_train, n_iterations=100)
    predictions = predict(X_test, weights)

    mse = mean_squared_error(y_test, predictions)
    assert mse > 0.00001
    assert mse < 1

    r2 = r2_score(y_test, predictions)
    assert r2 >= 0.9
    assert r2 < 1.0


def test_lasso_regression():
    """Test the lasso regression model."""
    X_train, X_test, y_train, y_test = regression_dataset()

    weights = lasso_regression(X_train, y_train, n_iterations=350)
    predictions = predict(X_test, weights)

    mse = mean_squared_error(y_test, predictions)
    assert mse > 0.00001
    assert mse < 1

    r2 = r2_score(y_test, predictions)
    assert r2 >= 0.9
    assert r2 < 1.0


def test_ridge_regression():
    """Test the ridge regression model."""
    X_train, X_test, y_train, y_test = regression_dataset()

    weights = ridge_regression(X_train, y_train, n_iterations=350)
    predictions = predict(X_test, weights)

    mse = mean_squared_error(y_test, predictions)
    assert mse > 0.00001
    assert mse < 1

    r2 = r2_score(y_test, predictions)
    assert r2 >= 0.9
    assert r2 < 1.0


def test_elastic_net():
    """Test the elastic net model."""
    X_train, X_test, y_train, y_test = regression_dataset()

    weights = elastic_net_regression(X_train, y_train, n_iterations=350)
    predictions = predict(X_test, weights)

    mse = mean_squared_error(y_test, predictions)
    assert mse > 0.00001
    assert mse < 1

    r2 = r2_score(y_test, predictions)
    assert r2 >= 0.9
    assert r2 < 1.0


def test_logistic_regression():
    """Test the logistic regression model."""
    X_train, X_test, y_train, y_test = classification_dataset()

    weights = logistic_regression(X_train, y_train, n_iterations=100)
    predictions = predict_logistic(X_test, weights)

    acc = accuracy_score(y_test, predictions)

    assert acc >= 0.9
    assert acc < 1.0

    f1 = f1_score(y_test, predictions)
    assert f1 >= 0.9
    assert f1 < 1.0
