import activation
import numpy as np

# relu activation function
class Relu(activation.Activation):

    def __init__(self):
        super(Relu, self).__init__()

    # nothing to be built
    # do not call explicitely
    def _build(self, X):
        pass

    # fwd pass
    # do not call explicitely
    # this method applies the relu nonlinearity to the input tensor
    def _forward(self, X):
        self.__mask = np.where(X > 0, 1, 0) # store if the element has grad
        return np.where(X > 0, X, 0)
    
    # backpropagation
    # must be called explicitely
    def backprop(self, grad):
        return grad*self.__mask # only pass grad to the previous layer if the element has grad
