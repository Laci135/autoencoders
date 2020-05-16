import modules

# abstract loss class
class Loss(modules.Module):

    def __init__(self):
        super(Loss, self).__init__()
    
    # fwd pass
    # checkpoints data
    # do not call explicitely
    def _forward(self, X):
        self.__last = X
        return X
    
    # backprop
    # nothing special happens, its just a loss function
    def backprop(self, grad):
        return grad

    # calculate loss mx using a specific loss formula
    # must be overridden
    # do not call explicitely
    def _calculate(self, X, Y=None):
        raise Exception("This method is abstract. The loss function is not defined.")

    # calculate grad mx using the derivative of the loss formula
    # must be overridden
    # do not call explicitely
    def _calculate_gradient(self, X, Y=None):
        raise Exception("This method is abstract. The gradient function is not defined.")
    
    # calculate total loss using a specific loss formula
    # must be overridden
    # do not call explicitely
    def _calculate_total(self, X, Y=None):
        raise Exception("This method is abstract. The total gradient function is not defined.")

    # calculates the total loss
    def total(self, Y=None):
        return self._calculate_total(self.__last, Y)

    # calculates loss
    def get(self, Y=None):
        return self._calculate(self.__last, Y)

    # calculates grad
    def get_grad(self, Y=None):
        return self._calculate_grad(self.__last, Y)

