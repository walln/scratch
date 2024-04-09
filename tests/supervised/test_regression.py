"""Test regression models."""

from scratch.supervised.regression import (
    ElasticNet,
    LassoRegression,
    LinearRegression,
    LogisticRegression,
    RidgeRegression,
)
from scratch.utils.logging import setup_logger
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = setup_logger()


def regression_dataset():
    """Return a regression dataset."""
    X, y = make_regression(n_samples=2000, n_features=1, noise=20, random_state=1)
    # X, y = load_diabetes(return_X_y=True)

    data_scaler = StandardScaler()
    target_scaler = MinMaxScaler()

    X = data_scaler.fit_transform(X)
    y = target_scaler.fit_transform(y.reshape(-1, 1))
    y = y.reshape(-1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, shuffle=False
    )

    return X_train, X_test, y_train, y_test


def test_linear_regression():
    """Test the linear regression model."""
    X_train, X_test, y_train, y_test = regression_dataset()

    model = LinearRegression(n_iterations=75)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    logger.info(f"MSE: {mse}")
    assert mse > 0.00001
    assert mse < 0.01

    r2 = r2_score(y_test, predictions)
    logger.info(f"R2 Score: {r2}")
    assert r2 >= 0.9
    assert r2 < 1.0


def test_lasso_regression():
    """Test the lasso regression model."""
    X_train, X_test, y_train, y_test = regression_dataset()

    model = LassoRegression(n_iterations=75)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    logger.info(f"MSE: {mse}")
    assert mse > 0.00001
    assert mse < 0.01

    r2 = r2_score(y_test, predictions)
    logger.info(f"R2 Score: {r2}")
    assert r2 >= 0.9
    assert r2 < 1.0


def test_ridge_regression():
    """Test the ridge regression model."""
    X_train, X_test, y_train, y_test = regression_dataset()

    model = RidgeRegression(n_iterations=75)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    logger.info(f"MSE: {mse}")
    assert mse > 0.00001
    assert mse < 0.01

    r2 = r2_score(y_test, predictions)
    logger.info(f"R2 Score: {r2}")
    assert r2 >= 0.9
    assert r2 < 1.0


def test_elastic_net():
    """Test the elastic net model."""
    X_train, X_test, y_train, y_test = regression_dataset()

    model = ElasticNet(n_iterations=5000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    logger.info(f"MSE: {mse}")
    assert mse > 0.00001
    assert mse < 0.01

    r2 = r2_score(y_test, predictions)
    logger.info(f"R2 Score: {r2}")
    assert r2 >= 0.9
    assert r2 < 1.0


def test_logistic_regression():
    """Test the logistic regression model."""
    X, y = make_classification(n_samples=2000, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, shuffle=False
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    logger.info(f"Accuracy score: {acc}")
    assert acc >= 0.9
    assert acc < 1.0

    f1 = f1_score(y_test, predictions)
    logger.info(f"F1 Score: {f1}")
    assert f1 >= 0.9
    assert f1 < 1.0
