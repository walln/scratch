"""Utility functions for data manipulation."""
import numpy as np


def diagonalize(x):
    """Converts a vector into an diagonal matrix.

    Args:
    ----
    x: the vector to convert into a diagonal matrix
    """
    m = np.zeros((len(x), len(x)))
    for i in range(len(m[0])):
        m[i, i] = x[i]
    return m
