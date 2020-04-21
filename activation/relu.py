import activation

class Relu(activation.Activation):

    def __init__(self):
        super(Relu, self).__init__()

    def _forward(self, X):
        self.__mask = np.where(X > 0, 1, 0)
        return np.where(X > 0, X, 0)

    def backprop(self, X):
        return X*mask
