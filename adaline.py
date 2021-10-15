import numpy as np
from model import BipolarActivation


class Adaline:
    def __init__(self, eta=0.05, error_threshold=0.3, weights_init_range=0.01, bias_init_range=0.01, bias_val=None):
        self.eta = eta
        self.activation = BipolarActivation()
        self.ERROR_THRESHOLD = error_threshold
        self.WEIGHTS_INIT_RANGE = weights_init_range

        if bias_val is not None:
            self.bias_ = bias_val
        else:
            self.bias_ = np.random.uniform(-bias_init_range, bias_init_range)

    def fit(self, X, y, max_epochs=100):
        self.costs = []
        self.w_ = np.random.uniform(-self.WEIGHTS_INIT_RANGE, self.WEIGHTS_INIT_RANGE, np.shape(X)[1])

        for epoch in range(max_epochs):
            errors = y - self.output(X)

            self.w_ += self.eta * np.dot(errors, X)
            self.bias_ += self.eta * errors.sum()
            self.costs.append((errors**2).sum() / len(y))

            if self.costs[-1] < self.ERROR_THRESHOLD:
                return epoch+1

        return max_epochs

    def output(self, X):
        return X.dot(self.w_.T) + self.bias_

    def predict(self, X):
        return self.activation.activate(self.output(X))
