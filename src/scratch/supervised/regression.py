"""Regression models for supervised learning."""

import math
from enum import Enum
from typing import Literal

import numpy as np
from scratch.activation_functions.sigmoid import Sigmoid
from scratch.utils.data import diagonalize
from scratch.utils.logging import setup_logger

logger = setup_logger()


class L1_Regularization:
    """Regularization for Lasso Regression."""

    def __init__(self, alpha):
        """Initialize regularization parameter.

        Args:
        ----
            alpha: regularization strength
        """
        self.alpha = alpha

    def __call__(self, w):
        """Calculate regularization term.

        Args:
        ----
            w: weights
        """
        return self.alpha * np.linalg.norm(w, 1)

    def grad(self, w):
        """Calculate gradient of regularization term.

        Args:
        ----
            w: weights
        """
        return self.alpha * np.sign(w)


class L2_Regularization:
    """Regularization for Ridge Regression."""

    def __init__(self, alpha):
        """Initialize regularization parameter.

        Args:
        ----
            alpha: regularization strength
        """
        self.alpha = alpha

    def __call__(self, w):
        """Calculate regularization term.

        Args:
        ----
        w: weights
        """
        return self.alpha * 0.5 * w.T.dot(w)

    def grad(self, w):
        """Calculate gradient of regularization term.

        Args:
        ----
            w: weights
        """
        return self.alpha * w


class Elastic_Regularization:
    """Regularization for Elastic Net using combined L1 and L2 regularization."""

    def __init__(self, alpha, l1_ratio=0.5):
        """Initialize regularization parameter.

        Args:
        ----
            alpha: regularization strength
            l1_ratio: ratio of L1 regularization
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def __call__(self, w):
        """Calculate regularization term.

        Args:
        ----
        w: weights
        """
        l1_contr = self.l1_ratio * np.linalg.norm(w, 1)
        l2_contr = (1 - self.l1_ratio) * 0.5 * w.T.dot(w)
        return self.alpha * (l1_contr + l2_contr)

    def grad(self, w):
        """Calculate gradient of regularization term.

        Args:
        ----
        w: weights
        """
        l1_contr = self.l1_ratio * np.sign(w)
        l2_contr = (1 - self.l1_ratio) * w
        return self.alpha * (l1_contr + l2_contr)


class OptimizationMethod(str, Enum):
    """Optimization methods for training regression models."""

    gradient_descent = "gradient_descent"
    least_squares = "least_squares"


class Regression:
    """Base regression model that fits a relationship between scalars X and Y.

    Args:
    ----
    n_iterations: float
        The number of training iterations the algorithm will take to attempt to
        optimize the model.
    learning_rate: float
        The rate of change of the model parameters at each iteration.
    """

    def __init__(self, n_iterations, learning_rate):
        """Initialize model parameters.

        Args:
        ----
            n_iterations: number of training iterations
            learning_rate: step length for updating model parameters
        """
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0

    def init_weights(self, n_features):
        """Randomly initialize the model weights between [-1/N, 1/N].

        Args:
        ----
        n_features: number of features
        """
        lim = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-lim, lim, (n_features,))

    def fit(self, X, y):
        """Fit the model to the training data.

        Args:
        ----
            X: training data
            y: target values
        """
        # Add static bias term
        X = np.insert(X, 0, 1, axis=1)
        self.init_weights(n_features=X.shape[1])
        self.training_errors = []

        for _ in range(self.n_iterations):
            y_pred = X.dot(self.w)
            # Calculate l2 loss
            mse = np.mean(0.5 * (y - y_pred) ** 2 + self.regularization(self.w))
            self.training_errors.append(mse)
            # Calculate the gradient of the l2 loss with respect to w
            grad_w = -(y - y_pred).dot(X) + self.regularization.grad(self.w)
            # Update the parameters of the weights
            self.w -= self.learning_rate * grad_w

    def predict(self, X):
        """Predict target values for the given data.

        Args:
        ----
            X: data to predict target values for
        """
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred


class LinearRegression(Regression):
    """Linear model that fits a relationship between scalars X and Y.

    Args:
    ----
    n_iterations: float
        The number of training iterations the algorithm will take to attempt to
        optimize the model.
    learning_rate: float
        The rate of change of the model parameters at each iteration.
    optimization: OptimizationMethod
        The optimization method to use when training the model. Either gradient
        descent or least squares.
    """

    def __init__(
        self,
        n_iterations=1000,
        learning_rate=0.001,
        optimization=OptimizationMethod.least_squares,
    ):
        """Initialize model parameters.

        Args:
        ----
            n_iterations: number of training iterations
            learning_rate: step length for updating model parameters
            optimization: optimization method
        """
        self.optimization = optimization
        # No regularization
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super().__init__(n_iterations=n_iterations, learning_rate=learning_rate)

    def fit(self, X, y):
        """Fit the model to the training data.

        Args:
        ----
            X: training data
        y: target values
        """
        if self.optimization == OptimizationMethod.least_squares:
            # Add static bias term
            X = np.insert(X, 0, 1, axis=1)
            # Calculate weights by least squares
            # https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse
            U, S, V = np.linalg.svd(X.T.dot(X))
            inv = V.dot(np.linalg.pinv(np.diag(S))).dot(U.T)
            self.w = inv.dot(X.T).dot(y)
        else:
            super().fit(X, y)


class LassoRegression(Regression):
    """Linear regression model with regularization.

    This model tries to balance the fit of
    the model with respect to the training data and the complexity of the model using
    l1 regularization.

    Args:
    ----
    learning_rate: The step length that will be used when updating the weights.
    n_iterations: The number of training iterations the algorithm will take to attempt
        to optimize the model.
    reg_factor: The amount of regularization and feature shrinkage.
    """

    def __init__(self, learning_rate=0.001, n_iterations=1000, reg_factor=0.5):
        """Initialize model parameters.

        Args:
        ----
            n_iterations: number of training iterations
            learning_rate: step length for updating model parameters
            reg_factor: regularization strength
        """
        self.regularization = L1_Regularization(alpha=reg_factor)
        super().__init__(n_iterations, learning_rate)

    def fit(self, X, y):
        """Fit the model to the training data.

        Args:
        ----
            X: training data
        y: target values
        """
        super().fit(X, y)

    def predict(self, X):
        """Predict target values for the given data.

        Args:
        ----
        X: data to predict target values for
        """
        return super().predict(X)


class RidgeRegression(Regression):
    """Linear regression model with regularization.

    This model tries to balance the fit
    of the model with respect to the training data and the complexity of the model
    using l2 regularization.

    Args:
    ----
    learning_rate: The step length that will be used when updating the weights.
    n_iterations: The number of training iterations the algorithm will take to attempt
        to optimize the model.
    reg_factor: The amount of regularization and feature shrinkage.
    """

    def __init__(self, learning_rate=0.001, n_iterations=1000, reg_factor=0.5):
        """Initialize model parameters.

        Args:
        ----
            n_iterations: number of training iterations
            learning_rate: step length for updating model parameters
            reg_factor: regularization strength
        """
        self.regularization = L2_Regularization(alpha=reg_factor)
        super().__init__(n_iterations, learning_rate)


class ElasticNet(Regression):
    """Regression using both L1 and L2 regularization.

    Args:
    ----------
    learning_rate: The step length that will be used when updating the weights
    n_iterations: The number of training iterations the algorithm will take to attempt
        to optimize the model.
    reg_factor: The amount of regularization and feature shrinkage
    l1_ratio: The contribution of L1 regularization in the combined regularization term
    """

    def __init__(
        self, learning_rate=0.001, n_iterations=1000, reg_factor=0.05, l1_ratio=0.5
    ):
        """Initialize model parameters.

        Args:
        ----
            n_iterations: number of training iterations
            learning_rate: step length for updating model parameters
            reg_factor: regularization strength
            l1_ratio: ratio of L1 regularization
        """
        self.regularization = Elastic_Regularization(
            alpha=reg_factor, l1_ratio=l1_ratio
        )

        super().__init__(n_iterations, learning_rate)

    def fit(self, X, y):
        """Fit the model to the training data.

        Args:
        ----
            X: training data
            y: target values
        """
        # X = normalize(polynomial_features(X, degree=self.degree))
        super().fit(X, y)

    def predict(self, X):
        """Predict target values for the given data.

        Args:
        ----
            X: data to predict target values for
        """
        # X = normalize(polynomial_features(X, degree=self.degree))
        return super().predict(X)


class LogisticRegression:
    """Logistic Regression classifier.

    Performs binary classification by estimating
    probability of an input belonging to a certain class utilizing a sigmoid
    activation.

    Args:
    ----
    learning_rate: The step length that will be used when updating the weights.
     n_iterations: The number of training iterations the algorithm will take to
        attempt to optimize the model.
    optimization: The optimization method to use when training the model.
        Either gradient descent or least squares.
    """

    def __init__(
        self,
        learning_rate=0.001,
        n_iterations=1000,
        optimization: Literal["gradient_descent", "least_squares"] = "gradient_descent",
    ):
        """Initialize model parameters.

        Args:
        ----
            n_iterations: number of training iterations
            learning_rate: step length for updating model parameters
            optimization: optimization method
        """
        self.params = None
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.optimization = optimization
        self.sigmoid = Sigmoid()

    def _initialize_parameters(self, X):
        n_features = np.shape(X)[1]
        # Initial parameters should be uniformly distributed between
        # [-1/sqrt(N), 1/sqrt(N)]
        lim = 1 / math.sqrt(n_features)
        self.params = np.random.uniform(-lim, lim, (n_features,))

    def fit(self, X, y):
        """Fit the model to the training data.

        Args:
        ----
            X: training data
            y: target values
        """
        self._initialize_parameters(X)
        for _ in range(self.n_iterations):
            y_pred = self.sigmoid(X.dot(self.params))
            if self.optimization == "gradient_descent":
                # Update parameters opposite to the direction of the loss gradient
                self.params -= self.learning_rate * -(y - y_pred).dot(X)
            elif self.optimization == "least_squares":
                # Diagonal of the activation gradient
                diag_gradient = diagonalize(self.sigmoid.gradient(X.dot(self.params)))
                self.params = (
                    np.linalg.pinv(X.T.dot(diag_gradient).dot(X))
                    .dot(X.T)
                    .dot(diag_gradient.dot(X).dot(self.params) + y - y_pred)
                )

    def predict(self, X):
        """Predict target values for the given data.

        Args:
        ----
        X: data to predict target values for
        """
        return np.round(self.sigmoid(X.dot(self.params))).astype(int)
