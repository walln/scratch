"""Regression models for supervised learning."""

from collections.abc import Callable
from enum import Enum
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp


class OptimizationMethod(str, Enum):
    """Optimization methods for regression.

    The optimization methods available are:
        - gradient_descent: The gradient descent algorithm.
        - least_squares: The least squares method.
    """

    gradient_descent = "gradient_descent"
    least_squares = "least_squares"


def l1_regularization(w: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """The L1 regularization function.

    The L1 regularization function is defined as:

            f(w) = alpha * ||w||_1

    Args:
        w: The weights of the model.
        alpha: The regularization factor.

    Returns:
        The L1 regularization value.
    """
    return alpha * jnp.linalg.norm(w, 1)


def l2_regularization(w: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """The L2 regularization function.

    The L2 regularization function is defined as:

            f(w) = alpha * 0.5 * w.T.dot(w)

    Args:
        w: The weights of the model.
        alpha: The regularization factor.

    Returns:
        The L2 regularization value.
    """
    return alpha * 0.5 * jnp.dot(w, w)


def elastic_regularization(
    w: jnp.ndarray, alpha: float, l1_ratio: float = 0.5
) -> jnp.ndarray:
    """The elastic net regularization function.

    The elastic net regularization function is defined as:

            f(w) = alpha * (l1_ratio * ||w||_1 + (1 - l1_ratio) * 0.5 * w.T.dot(w))

    Args:
        w: The weights of the model.
        alpha: The regularization factor.
        l1_ratio: The ratio of L1 regularization in the model.

    Returns:
        The elastic net regularization value.
    """
    l1_contr = l1_ratio * jnp.linalg.norm(w, 1)
    l2_contr = (1 - l1_ratio) * 0.5 * w.T.dot(w)
    return alpha * (l1_contr + l2_contr)


def l1_grad(w: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """The gradient of the L1 regularization function.

    The L1 regularization function is defined as:

            f(w) = alpha * ||w||_1

    Args:
        w: The weights of the model.
        alpha: The regularization factor.

    Returns:
        The gradient of the L1 regularization function.
    """
    return alpha * jnp.sign(w)


def l2_grad(w: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """The gradient of the L2 regularization function.

    The L2 regularization function is defined as:

                f(w) = alpha * w

    Args:
        w: The weights of the model.
        alpha: The regularization factor.

    Returns:
        The gradient of the L2 regularization function.
    """
    return alpha * w


def elastic_grad(w: jnp.ndarray, alpha: float, l1_ratio: float = 0.5) -> jnp.ndarray:
    """The gradient of the elastic net regularization function.

    The elastic net regularization function is defined as:

                f(w) = alpha * (l1_ratio * ||w||_1 + (1 - l1_ratio) * 0.5 * w.T.dot(w))

    Args:
        w: The weights of the model.
        alpha: The regularization factor.
        l1_ratio: The ratio of L1 regularization in the model.

    Returns:
        The gradient of the elastic net regularization function.
    """
    l1_contr = l1_ratio * jnp.sign(w)
    l2_contr = (1 - l1_ratio) * w
    return alpha * (l1_contr + l2_contr)


def init_weights(n_features: int, seed: int = 0) -> jnp.ndarray:
    """Initialize the weights of the model.

    The weights are initialized using a uniform distribution with a range of
    [-lim, lim], where lim is calculated as 1 / sqrt(n_features).

    Args:
        n_features: The number of features.
        seed: The random seed.

    Returns:
        The initialized weights.
    """
    lim = 1 / jnp.sqrt(n_features)
    return jax.random.uniform(
        jax.random.PRNGKey(seed), shape=(n_features,), minval=-lim, maxval=lim
    )


def add_bias(X: jnp.ndarray) -> jnp.ndarray:
    """Add a bias term to the input features.

    Args:
        X: The input features.

    Returns:
        The input features with a bias term.
    """
    return jnp.hstack((jnp.ones((X.shape[0], 1)), X))


def regression_step(
    w: jnp.ndarray,
    X: jnp.ndarray,
    y: jnp.ndarray,
    learning_rate: float,
    regularization: Callable[[jnp.ndarray], jnp.ndarray],
    regularization_grad: Callable[[jnp.ndarray], jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Perform a single step of regression.

    Args:
        w: The weights of the model.
        X: The input features.
        y: The target labels.
        learning_rate: The learning rate.
        regularization: The regularization function.
        regularization_grad: The gradient of the regularization function.

    Returns:
        The updated weights and the mean squared error.
    """
    y_pred = X.dot(w)
    mse = jnp.mean(0.5 * (y - y_pred) ** 2 + regularization(w[1:]))
    grad_w = -X.T.dot(y - y_pred) / X.shape[0]
    reg_grad = jnp.concatenate([jnp.zeros(1), regularization_grad(w[1:])])
    grad_w += reg_grad
    w -= learning_rate * grad_w
    return w, mse


def fit_regression(
    X: jnp.ndarray,
    y: jnp.ndarray,
    n_iterations: int = 1000,
    learning_rate: float = 0.001,
    regularization: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: jnp.zeros(0),
    regularization_grad: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: jnp.zeros(0),
    optimization: OptimizationMethod = OptimizationMethod.gradient_descent,
    seed: int = 0,
) -> tuple[jnp.ndarray, list]:
    """Fit a regression model.

    Args:
        X: The input features.
        y: The target labels.
        n_iterations: The number of iterations to run.
        learning_rate: The learning rate.
        regularization: The regularization function.
        regularization_grad: The gradient of the regularization function.
        optimization: The optimization method to use.
        seed: The random seed.

    Returns:
        The weights of the model and the training errors.
    """
    X = add_bias(X)
    w = init_weights(X.shape[1], seed)
    training_errors = []

    if optimization == OptimizationMethod.least_squares:
        U, S, Vt = jnp.linalg.svd(X, full_matrices=False)
        S_inv = jnp.diag(1 / S)
        w = Vt.T.dot(S_inv).dot(U.T).dot(y)
    else:
        for _ in range(n_iterations):
            w, mse = regression_step(
                w, X, y, learning_rate, regularization, regularization_grad
            )
            training_errors.append(mse)

    return w, training_errors


def predict(X: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Predict the target variable using the input features and the model weights.

    Args:
        X: The input features.
        w: The weights of the model.

    Returns:
        The predicted target variable
    """
    X = add_bias(X)
    return X.dot(w)


# Specific regression types
def linear_regression(
    X: jnp.ndarray,
    y: jnp.ndarray,
    n_iterations: int = 1000,
    learning_rate: float = 0.001,
    optimization: OptimizationMethod = OptimizationMethod.least_squares,
    seed: int = 0,
) -> jnp.ndarray:
    """Fit a linear regression model.

    Linear regression is a linear model that assumes a linear relationship between the
    input features and the target variable. The model is trained using the maximum
    likelihood estimation, which is equivalent to minimizing the mean squared error
    loss. The weights of the model are updated using the gradient descent algorithm.

    Args:
        X: The input features.
        y: The target labels.
        n_iterations: The number of iterations to run.
        learning_rate: The learning rate.
        optimization: The optimization method to use.
        seed: The random seed.

    Returns:
        The weights of the model.
    """
    return fit_regression(
        X, y, n_iterations, learning_rate, optimization=optimization, seed=seed
    )[0]


def lasso_regression(
    X: jnp.ndarray,
    y: jnp.ndarray,
    n_iterations: int = 1000,
    learning_rate: float = 0.01,
    reg_factor: float = 0.01,
    seed: int = 0,
) -> jnp.ndarray:
    """Fit a lasso regression model.

    Lasso regression is a linear regression model that uses the L1 penalty to
    regularize the weights of the model. The model is trained using the maximum
    likelihood estimation, which is equivalent to minimizing the binary cross-entropy
    loss. The weights of the model are updated using the gradient descent algorithm.

    Args:
        X: The input features.
        y: The target labels.
        n_iterations: The number of iterations to run.
        learning_rate: The learning rate.
        reg_factor: The regularization factor.
        seed: The random seed.

    Returns:
        The weights of the model.
    """
    return fit_regression(
        X,
        y,
        n_iterations,
        learning_rate,
        regularization=partial(l1_regularization, alpha=reg_factor),
        regularization_grad=partial(l1_grad, alpha=reg_factor),
        seed=seed,
    )[0]


def ridge_regression(
    X: jnp.ndarray,
    y: jnp.ndarray,
    n_iterations: int = 1000,
    learning_rate: float = 0.01,
    reg_factor: float = 0.01,
    seed: int = 0,
) -> jnp.ndarray:
    """Fit a ridge regression model.

    Ridge regression is a linear regression model that uses the L2 penalty to
    regularize the weights of the model. The model is trained using the maximum
    likelihood estimation, which is equivalent to minimizing the binary cross-entropy
    loss. The weights of the model are updated using the gradient descent algorithm.

    Args:
        X: The input features.
        y: The target labels.
        n_iterations: The number of iterations to run.
        learning_rate: The learning rate.
        reg_factor: The regularization factor.
        seed: The random seed.

    Returns:
        The weights of the model.
    """
    return fit_regression(
        X,
        y,
        n_iterations,
        learning_rate,
        regularization=partial(l2_regularization, alpha=reg_factor),
        regularization_grad=partial(l2_grad, alpha=reg_factor),
        seed=seed,
    )[0]


def elastic_net_regression(
    X: jnp.ndarray,
    y: jnp.ndarray,
    n_iterations: int = 1000,
    learning_rate: float = 0.01,
    reg_factor: float = 0.01,
    l1_ratio: float = 0.5,
    seed: int = 0,
) -> jnp.ndarray:
    """Fit an elastic net regression model.

    Elastic net is a linear regression model that combines the L1 and L2 penalties of
    the Lasso and Ridge regression models. The model is trained using the maximum
    likelihood estimation, which is equivalent to minimizing the binary cross-entropy
    loss. The weights of the model are updated using the gradient descent algorithm.

    Args:
        X: The input features.
        y: The target labels.
        n_iterations: The number of iterations to run.
        learning_rate: The learning rate.
        reg_factor: The regularization factor.
        l1_ratio: The ratio of L1 regularization in the model.
        seed: The random seed.

    Returns:
        The weights of the model.
    """
    return fit_regression(
        X,
        y,
        n_iterations,
        learning_rate,
        regularization=partial(
            elastic_regularization, alpha=reg_factor, l1_ratio=l1_ratio
        ),
        regularization_grad=partial(elastic_grad, alpha=reg_factor, l1_ratio=l1_ratio),
        seed=seed,
    )[0]


def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """The core logistic function or sigmoid operation.

    Args:
        x: The input value.

    Returns:
        The output value.
    """
    return 1 / (1 + jnp.exp(-x))


@jax.jit
def logistic_regression_step(
    w: jnp.ndarray,
    X: jnp.ndarray,
    y: jnp.ndarray,
    learning_rate: float,
) -> jnp.ndarray:
    """Perform a single step of logistic regression.

    Args:
        w: The weights of the model.
        X: The input features.
        y: The target labels.
        learning_rate: The learning rate.

    Returns:
        The updated weights.
    """
    y_pred = sigmoid(X.dot(w))
    grad_w = -(y - y_pred).dot(X)
    w -= learning_rate * grad_w
    return w


def logistic_regression(
    X: jnp.ndarray,
    y: jnp.ndarray,
    n_iterations: int = 1000,
    learning_rate: float = 0.001,
    optimization: Literal["gradient_descent", "least_squares"] = "gradient_descent",
    seed: int = 0,
) -> jnp.ndarray:
    """Fit a logistic regression model.

    Logistic regression is a classification algorithm used to assign observations to a
    discrete set of classes. It is a linear model that uses the logistic function to
    model the probability that a given input belongs to a particular class. The logistic
    function that maps any real-valued number into the range [0, 1] is defined as:

            f(x) = 1 / (1 + e^(-x))

    The model is trained using the maximum likelihood estimation, which is equivalent to
    minimizing the binary cross-entropy loss. The weights of the model are updated using
    the gradient descent algorithm.

    Args:
        X: The input features.
        y: The target labels.
        n_iterations: The number of iterations to run.
        learning_rate: The learning rate.
        optimization: The optimization method to use.
        seed: The random seed.

    Returns:
        The weights of the model.
    """
    w = init_weights(X.shape[1], seed)

    if optimization == "gradient_descent":
        for _ in range(n_iterations):
            w = logistic_regression_step(w, X, y, learning_rate)
    elif optimization == "least_squares":
        for _ in range(n_iterations):
            y_pred = sigmoid(X.dot(w))
            sigmoid_grad = jax.grad(sigmoid)
            diag_gradient = jnp.diag(sigmoid_grad(X.dot(w)))
            w = (
                jnp.linalg.pinv(X.T.dot(diag_gradient).dot(X))
                .dot(X.T)
                .dot(diag_gradient.dot(X).dot(w) + y - y_pred)
            )

    return w


def predict_logistic(X: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Predict the class of each sample in X.

    Args:
        X: The input features.
        w: The weights of the model.

    Returns:
        The predicted class for each sample.
    """
    return jnp.round(sigmoid(X.dot(w))).astype(int)
