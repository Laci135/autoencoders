import modules

class Loss(modules.Module):

    def __init__(self):
        super(Loss, self).__init__()

    def _forward(self, X):
        self.__last = X
        return X

    def backprop(self, grad):
        return grad

    def _calculate(self, X, Y=None):
        raise Exception("This method is abstract. The loss function is not defined.")

    def _calculate_gradient(self, X, Y=None):
        raise Exception("This method is abstract. The gradient function is not defined.")
    
    def _calculate_total(self, X, Y=None):
        raise Exception("This method is abstract. The total gradient function is not defined.")

    def total(self, Y=None):
        return self._calculate_total(self.__last, Y)

    def get(self, Y=None):
        return self._calculate(self.__last, Y)

    def get_grad(self, Y=None):
        return self._calculate_grad(self.__last, Y)

