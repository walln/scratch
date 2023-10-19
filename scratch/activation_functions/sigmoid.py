import numpy as np


class Sigmoid:
    def __call__(self, theta):
        return 1 / (1 + np.exp(-theta))

    def gradient(self, theta):
        return self.__call__(theta) * (1 - self.__call__(theta))
