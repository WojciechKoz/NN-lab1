import numpy as np
from model import BipolarActivation, UnipolarActivation


class Perceptron:
    def __init__(self, eta=0.05, bipolar=True, weights_init_range=0.01, bias_init_range=0.01, bias_val=None):
        self.eta = eta
        self.activation = BipolarActivation() if bipolar else UnipolarActivation()
        self.WEIGHTS_INIT_RANGE = weights_init_range

        if bias_val is not None:
            self.bias_ = bias_val
        else:
            self.bias_ = np.random.uniform(-bias_init_range, bias_init_range)

    def fit(self, X, y, max_epochs=100):
        self.w_ = np.random.uniform(-self.WEIGHTS_INIT_RANGE, self.WEIGHTS_INIT_RANGE, np.shape(X)[1])

        for epoch in range(max_epochs):
            errors = y - self.predict(X)

            self.w_ += self.eta * np.dot(errors, X)
            self.bias_ += self.eta * errors.sum()

            if not np.any(errors):  # if all errors are equal to 0 - stop
                return epoch+1
        return max_epochs

    def predict(self, X):
        return self.activation.activate(X.dot(self.w_.T) + self.bias_)


