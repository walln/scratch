"""Sigmoid activation function."""

import numpy as np


class Sigmoid:
    """Sigmoid activation function."""

    def __call__(self, theta):
        """Return the sigmoid of theta.

        Args:
        ----
            theta: the input to the sigmoid function
        """
        return 1 / (1 + np.exp(-theta))

    def gradient(self, theta):
        """Return the gradient of the sigmoid function.

        Args:
        ----
        theta: the input to the sigmoid function
        """
        return self.__call__(theta) * (1 - self.__call__(theta))
