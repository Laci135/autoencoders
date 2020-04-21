import modules
import numpy as np
import math

class Maxpool(modules.Module):

    def __init__(self, n):
        super(Maxpool, self).__init__()
        self.n = n

    def _build(self, X):
        self.H = X.shape[1]
        self.W = X.shape[2] 
        self.I = X.shape[3]

        
    def __readmax(self, data, at):
        max = None
        for h in range(self.n):
            for w in range(self.n):
                coords = (at[0], at[1]*self.n + h, at[2]*self.n + w, at[3])
                if coords[1] < self.H and coords[2] < self.W:
                    val = data[coords]
                    if max is None or val > max:
                        max = val
                        self.__mask[at] = (h, w)
        return max

    def _forward(self, X):
        self.B = X.shape[0]
        assert X.shape == (self.B, self.H, self.W, self.I), f"Maxpool module: Please fix input dimensions: {X.shape} -> {(self.B, self.H, self.W, self.I)}"

        self.HH = math.ceil(self.H/self.n)
        self.WW = math.ceil(self.W/self.n)

        out = np.zeros((self.B, self.HH, self.WW, self.I), dtype=float)
        self.__mask = np.zeros((self.B, self.HH, self.WW, self.I), dtype=(int, 2))

        for b in range(self.B):
            for h in range(self.HH):
                for w in range(self.WW):
                    for i in range(self.I):
                        max = self.__readmax(X, (b, h, w, i))
                        out[b, h, w, i] = max

        return out

    def backprop(self, grad):
        assert grad.shape == (self.B, self.HH, self.WW, self.I), f"Maxpool module: Wrong backprop dimensions. Fix: {grad.shape} -> {(self.B, self.HH, self.WW, self.I)}"
        
        out = np.zeros((self.B, self.H, self.W, self.I))

        for b in range(self.B):
            for h in range(self.HH):
                for w in range(self.WW):
                    for i in range(self.I):
                        at = (b, h, w, i)
                        val = grad[at]
                        where = self.__mask[at]
                        out[b, h*self.n+where[0], w*self.n+where[1], i] = self.n*self.n*val
        return out

        
