class Module:
    built: bool

    def _build(self, X):
        raise Exception("This method is abstract. Module does not define a build method.")

    def _forward(self, X):
        raise Exception("This method is abstract. Module does not define a forward method.")

    def __call__(self, X):
        if not self.built:
             self._build()
        self._forward(X)

    def backprop(self, lr, grad=None):
        raise Exception("This method is abstract. Module does not define backpropagation.")
