class Module:

    def __init__(self):
        self.__built = False

    # this method builds the module, descendants must override int
    # do not call explicitely
    def _build(self, X):
        raise Exception("This method is abstract. Module does not define a build method.")

    # fwd step, descendants must override int
    # do not call explicitely
    def _forward(self, X):
        raise Exception("This method is abstract. Module does not define a forward method.")

    # build module if not built and perform fwd pass
    def __call__(self, X):
        if not self.__built:
            self._build(X)
            self.__built = True
        
        return self._forward(X)
    
    # backprop, descendants must override
    # must call explicitely
    def backprop(self, grad=None):
        raise Exception("This method is abstract. Module does not define backpropagation.")
