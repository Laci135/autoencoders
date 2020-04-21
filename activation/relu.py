import activation
import numpy as np

class Relu(activation.Activation):

    def __init__(self):
        super(Relu, self).__init__()

    def _build(self, X):
        pass

    def _forward(self, X):
        self.__mask = np.where(X > 0, 1, 0)
        return np.where(X > 0, X, 0)

    def backprop(self, grad):
        return grad*self.__mask
