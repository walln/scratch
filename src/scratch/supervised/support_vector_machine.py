"""Support Vector Machine (SVM) Implementation.

This script provides an implementation of a Support Vector Machine (SVM). SVM is a
supervised machine learning algorithm that can be used for both classification and
regression challenges. However, it is mostly used in classification problems.
The goal of the SVM algorithm is to find a hyperplane in an N-dimensional space that
distinctly classifies the data points.

References:
- Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine learning, 20(3),
  273-297.
  Available at: https://link.springer.com/article/10.1007/BF00994018

"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from jax import grad


def linear_kernel(x1: jnp.ndarray, x2: jnp.ndarray):
    """Linear kernel function.

    The linear kernel is the simplest kernel function. It is given by the inner product
    of the input vectors.

    Args:
        x1: An array representing the first input vector.
        x2: An array representing the second input vector.

    Returns:
        The inner product of the input vectors.
    """
    return jnp.dot(x1, x2)


def polynomial_kernel(
    x1: jnp.ndarray, x2: jnp.ndarray, degree: int = 3, coef0: int = 1
):
    """Polynomial kernel function.

    The polynomial kernel is a non-stationary kernel. It is well suited for problems
    where all the training data is normalized. It is defined as:

        K(x, y) = (x^T y + coef0)^degree

    Args:
        x1: An array representing the first input vector.
        x2: An array representing the second input vector.
        degree: The degree of the polynomial kernel function.
        coef0: The constant term in the polynomial kernel function.

    Returns:
        The polynomial kernel function
    """
    return (jnp.dot(x1, x2) + coef0) ** degree


def rbf_kernel(x1: jnp.ndarray, x2: jnp.ndarray, gamma: float = 1.0):
    """Radial Basis Function (RBF) kernel function.

    The Radial Basis Function (RBF) kernel is a stationary kernel. It is defined as:

        K(x, y) = exp(-gamma * ||x - y||^2)

    Args:
        x1: An array representing the first input vector.
        x2: An array representing the second input vector.
        gamma: The gamma parameter of the RBF kernel function.

    Returns:
        The RBF kernel function.
    """
    return jnp.exp(-gamma * jnp.sum((x1 - x2) ** 2))


def svm_classification_loss(
    params: jnp.ndarray, kernel: Callable, X: jnp.ndarray, y: jnp.ndarray, C
):
    """Support Vector Machine (SVM) classification loss function.

    The SVM loss function is defined as:

        L(w, b) = 0.5 * ||w||^2 + C * sum(max(0, 1 - y_i(w^T x_i + b)))

    Args:
        params: The parameters of the SVM model.
        kernel: The kernel function to use.
        X: The input data.
        y: The target labels.
        C: The regularization parameter.

    Returns:
        The SVM loss.
    """
    w, b = split_weights_and_bias(params)
    kernel_matrix = jax.vmap(lambda x1: jax.vmap(lambda x2: kernel(x1, x2))(X))(X)
    margins = 1 - y * (jnp.dot(kernel_matrix, w) + b)
    hinge_loss = jnp.maximum(0, margins)
    return 0.5 * jnp.dot(w, w) + C * jnp.sum(hinge_loss)


def svm_regression_loss(
    params: jnp.ndarray,
    kernel: Callable,
    X: jnp.ndarray,
    y: jnp.ndarray,
    C: float,
    epsilon: float = 0.1,
):
    """Support Vector Machine (SVM) regression loss function.

    The SVM regression loss function is defined as:

        L(w, b) = 0.5 * ||w||^2 + C * sum(max(0, |y_i - (w^T x_i + b)| - epsilon))

    Args:
        params: The parameters of the SVM model.
        kernel: The kernel function to use.
        X: The input data.
        y: The target labels.
        C: The regularization parameter.
        epsilon: The epsilon parameter of the SVM regression loss.

    Returns:
        The SVM regression loss.
    """
    w, b = split_weights_and_bias(params)
    kernel_matrix = jax.vmap(lambda x1: jax.vmap(lambda x2: kernel(x1, x2))(X))(X)
    predictions = jnp.dot(kernel_matrix, w) + b
    errors = jnp.abs(y - predictions) - epsilon
    hinge_loss = jnp.maximum(0, errors)
    return 0.5 * jnp.dot(w, w) + C * jnp.sum(hinge_loss)


def predict_classification(params: jnp.ndarray, kernel: Callable, X: jnp.ndarray):
    """Predict function for the SVM classification model.

    Args:
        params: The parameters of the SVM model.
        kernel: The kernel function to use.
        X: The input data.

    Returns:
        The predicted labels.
    """
    w, b = split_weights_and_bias(params)
    kernel_matrix = jax.vmap(lambda x1: jax.vmap(lambda x2: kernel(x1, x2))(X))(X)
    return jnp.sign(jnp.dot(kernel_matrix, w) + b)


def predict_regression(params: jnp.ndarray, kernel: Callable, X: jnp.ndarray):
    """Predict function for the SVM regression model.

    Args:
        params: The parameters of the SVM model.
        kernel: The kernel function to use.
        X: The input data.

    Returns:
        The predicted values.
    """
    w, b = split_weights_and_bias(params)
    kernel_matrix = jax.vmap(lambda x1: jax.vmap(lambda x2: kernel(x1, x2))(X))(X)
    return jnp.dot(kernel_matrix, w) + b


def split_weights_and_bias(params: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Split the weights and bias from the parameters.

    Args:
        params: The parameters of the SVM model.

    Returns:
        The weights and bias.
    """
    return params[:-1], params[-1]


def concat_weights_and_bias(W: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Concatenate the weights and bias into a single parameter vector.

    Args:
        W: The weights of the SVM model.
        b: The bias of the SVM model.

    Returns:
        The parameters of the SVM model.
    """
    return jnp.concatenate([W, b])


def support_vector_machine_classifier(
    X: jnp.ndarray,
    y: jnp.ndarray,
    kernel: Callable = linear_kernel,
    C: float = 1.0,
    learning_rate: float = 1e-3,
    num_epochs: int = 1000,
):
    """Train a Support Vector Machine (SVM) classification model.

    Args:
        X: The input data.
        y: The target labels.
        kernel: The kernel function to use.
        C: The regularization parameter.
        learning_rate: The learning rate of the optimizer.
        num_epochs: The number of training epochs.

    Returns:
        The parameters of the SVM model.
    """
    # Initialize parameters
    w = jnp.zeros(X.shape[0])
    b = jnp.zeros(1)

    params = concat_weights_and_bias(w, b)

    # Define the optimizer
    optimizer = optax.sgd(learning_rate)
    opt_state = optimizer.init(params)

    # Define the update function
    def update(params, opt_state, X, y, kernel, C):
        grads = grad(svm_classification_loss)(params, kernel, X, y, C)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    # Training loop
    for _ in range(num_epochs):
        params, opt_state = update(params, opt_state, X, y, kernel, C)

    return params


def support_vector_machine_regressor(
    X: jnp.ndarray,
    y: jnp.ndarray,
    kernel: Callable = linear_kernel,
    C: float = 1.0,
    epsilon: float = 0.1,
    learning_rate: float = 1e-3,
    num_epochs: int = 1000,
):
    """Train a Support Vector Machine (SVM) regression model.

    Args:
        X: The input data.
        y: The target labels.
        kernel: The kernel function to use.
        C: The regularization parameter.
        epsilon: The epsilon parameter in the epsilon-insensitive loss.
        learning_rate: The learning rate of the optimizer.
        num_epochs: The number of training epochs.

    Returns:
        The parameters of the SVM regression model.
    """
    # Initialize parameters
    w = jnp.zeros(X.shape[0])
    b = jnp.zeros(1)

    params = concat_weights_and_bias(w, b)

    # Define the optimizer
    optimizer = optax.sgd(learning_rate)
    opt_state = optimizer.init(params)

    # Define the update function
    def update(params, opt_state, X, y, kernel, C, epsilon):
        grads = grad(svm_regression_loss)(params, kernel, X, y, C, epsilon)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    # Training loop
    for _ in range(num_epochs):
        params, opt_state = update(params, opt_state, X, y, kernel, C, epsilon)

    return params
