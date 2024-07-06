"""K-Nearest Neighbors (KNN) Implementation.

This script provides an implementation of the K-Nearest Neighbors (KNN) algorithm.
KNN is a simple, supervised machine learning algorithm that can be used for both
classification and regression tasks. The fundamental concept behind KNN is that
similar data points are close to each other. For classification, KNN assigns the
class of a data point based on the majority class of its k nearest neighbors. For
regression, it predicts the value of a data point based on the average of the values
of its k nearest neighbors.

The KNN algorithm is non-parametric and lazy, meaning it makes no assumptions about
the underlying data distribution and does not learn a discriminative function from
the training data. Instead, it stores all the training data and performs computation
only during the prediction phase.

References:
- Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. IEEE
  Transactions on Information Theory, 13(1), 21-27.
  Available at: https://ieeexplore.ieee.org/document/1053964

"""

import jax.numpy as jnp


def knn(X_train: jnp.ndarray, y_train: jnp.ndarray, X_test: jnp.ndarray, k: int = 3):
    """K-Nearest Neighbors (KNN) algorithm.

    KNN is a simple, supervised machine learning algorithm that can be used for both
    classification and regression tasks. For classification, KNN assigns the class of
    a data point based on the majority class of its k nearest neighbors. For regression,
    it predicts the value of a data point based on the average of the values of its k
    nearest neighbors.

    Args:
        X_train: An array representing the features of the training data.
        y_train: An array representing the labels of the training data.
        X_test: An array representing the features of the test data.
        k: An integer representing the number of neighbors to consider.

    Returns:
        An array of predicted labels for the test data.
    """
    # Compute the pairwise Euclidean distances between the test and training data
    distances = jnp.linalg.norm(X_train[:, None] - X_test, axis=2)

    # Find the indices of the k nearest neighbors for each test data point
    nearest_indices = jnp.argsort(distances, axis=0)[:k]

    # Get the labels of the k nearest neighbors
    nearest_labels = y_train[nearest_indices]

    # Compute the majority class label for each test data point
    y_pred = jnp.array(
        [jnp.bincount(nearest_labels[:, i]).argmax() for i in range(X_test.shape[0])]
    )

    return y_pred
