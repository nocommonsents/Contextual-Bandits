__author__ = 'bixlermike'

import numpy as np

class LinearScenario:
    def __init__(self, rewards, k, sparsity, scale):
        self.rewards = rewards
        self.k = k
        self.sparsity = sparsity
        self.scale = scale

    @property
    def _n(self):
        return len(self.rewards)

    def _make_action(self):
        x = np.random.uniform(0.0, 1.0, self._n) >= self.sparsity
        return x

    def get_actions(self):
        return [self._make_action() for _ in xrange(self.k)]

    def evaluate_action(self, action, scale=None):
        if scale is None:
            scale = self.scale
        if scale == 0.0:
            noise = 0
        else:
            noise = np.random.normal(0.0, scale=scale)
        return np.dot(action, self.rewards) + noise
