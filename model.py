import numpy as np


class Activation:
    def activate(self, X):
        pass


class BipolarActivation(Activation):
    def activate(self, X):
        return np.where(X > 0, 1, -1)


class UnipolarActivation(Activation):
    def activate(self, X):
        return np.where(X > 0, 1, 0)
