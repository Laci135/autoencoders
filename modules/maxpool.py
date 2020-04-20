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
                coords = (at[0]*self.n + h, at[1]*self.n + w)
                max = None
                if coords[0] and coords[1] < self.W:
                    val = data[coords]
                    if max is None or val > max:
                        max = val
                        self.mask[at] = (h, w)
                   
    def _forward(self, X):
        self.B = X.shape[0]
        assert X.shape == (self.B, self.H, self.W, self.I), f"Maxpool module: Please fix input dimensions: {X.shape} -> {(self.B, self.H, self.W, self.I)}"
       
        out = np.zeros((math.ceil(self.H/self.n), math.ceil(self.W/self.n)), dtype=float)

        self.mask = np.zeros((math.ceil(self.H/self.n), math.ceil(self.W/self.n)), dtype=(int, 2))

        for b in range(self.B):
            for i in range(self.I):
                for h in range(self.H):
                    for w in range(self.W):
                         max = self.__readmax(X, (b, i, h, w))
                         out[b, i, h, w] = max
    
   
    def __fillgrad(self, data, at, val):
        h, w = mask[at]

        data[at[0], at[1], at[2]*self.n+h, at[3]*self.n+w]

    def backprop(self, lr, grad):
        assert loss.shape == (self.B, math.ceil(self.H/self.n), math.ceil(self.H/self.n), self.I), f"Maxpool module: Wrong backprop dimensions. Fix: {loss.shape} -> {(self.B, math.ceil(self.H/self.n), math.ceil(self.H/self.n), self.I)}"

        out = np.zeros((self.B, self.H, self.W, self.I))

        for b in range(self.B):
            for h in range(self.H):
                for w in range(self.W):
                    for i in range(self.I):
                        val = grad[at]
                        __fillgrad(out, at, val)
        
        return out

        
