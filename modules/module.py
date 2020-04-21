class Module:

    def __init__(self):
        self.__built = False

    def _build(self, X):
        raise Exception("This method is abstract. Module does not define a build method.")

    def _forward(self, X):
        raise Exception("This method is abstract. Module does not define a forward method.")

    def __call__(self, X):
        if not self.__built:
            self._build(X)
            self.__built = True
        
        return self._forward(X)

    def backprop(self, grad=None):
        raise Exception("This method is abstract. Module does not define backpropagation.")
