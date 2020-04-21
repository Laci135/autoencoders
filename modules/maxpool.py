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
        for h in range(self.n):
            for w in range(self.n):
                coords = (at[0], at[1]*self.n + h, at[2]*self.n + w, at[3])
                max = None
                if coords[1] < self.H and coords[2] < self.W:
                    val = data[coords]
                    if max is None or val > max:
                        max = val
                        self.mask[at] = (h, w)
                   
    def _forward(self, X):
        self.B = X.shape[0]
        assert X.shape == (self.B, self.H, self.W, self.I), f"Maxpool module: Please fix input dimensions: {X.shape} -> {(self.B, self.H, self.W, self.I)}"

        self.HH = math.ceil(self.H/self.n)
        self.WW = math.ceil(self.W/self.n)

        out = np.zeros((self.B, self.HH, self.WW, self.I), dtype=float)
        self.mask = np.zeros((self.B, self.HH, self.WW, self.I), dtype=(int, 2))

        for b in range(self.B):
            for h in range(self.HH):
                for w in range(self.WW):
                    for i in range(self.I):
                        max = self.__readmax(X, (b, h, w, i))
                        out[b, h, w, i] = max

        return out
   
    def __fillgrad(self, data, at, val):
        h, w = self.mask[at]
        data[at[0], at[1]*self.n+h, at[2]*self.n+w, at[3]]

    def backprop(self, lr, grad):
        assert grad.shape == (self.B, self.HH, self.WW, self.I), f"Maxpool module: Wrong backprop dimensions. Fix: {grad.shape} -> {(self.B, self.HH, self.WW, self.I)}"
        
        out = np.zeros((self.B, self.H, self.W, self.I))

        for b in range(self.B):
            for h in range(self.HH):
                for w in range(self.WW):
                    for i in range(self.I):
                        at = (b, h, w, i)
                        val = grad[at]
                        self.__fillgrad(out, at, val)
        
        return out

        
