import modules

class Loss(modules.Module):

    def _forward(self, X):
        _calculate(X)
        _calulate_grad(X)
        return X

    def backprop(self, lr, loss):
        return X

    def _calculate(self, X, Y):
        raise Exception("This method is abstract. The loss function is not defined.")

    def _calculate_gradient(self, X, Y):
        raise Exception("This method is abstract. The gradient function is not defined.")

    def total(self):
        raise Exception("This method is abstract. The printable form of the loss function cannot be calculated")
    
    def get(self):
        return __loss

    def _set(self, loss):
        __loss = loss

    def _set_grad(self, grad):
        __grad = grad
