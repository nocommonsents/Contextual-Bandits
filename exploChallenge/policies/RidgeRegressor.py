import numpy as np

class RidgeRegressor:
    def __init__(self, A0, b0):
        (m, n) = A0.shape
        assert m == n
        self.m = m
        self.A = A0
        self.b = b0
        self._A_inv = None

    def update(self, x, y):
        assert x.shape == (self.m, )
        self.A += np.outer(x, x)
        self.b += x * y
        self._A_inv = None

    @property
    def A_inv(self):
        if self._A_inv is None:
            self._A_inv = np.linalg.inv(self.A)
        return self._A_inv

    def predict(self, x):
        theta = np.dot(self.A_inv, self.b)
        return np.dot(theta, x)

    def variance_multiplier(self, x):
        return np.dot(x, np.dot(self.A_inv, x)) ** 0.5