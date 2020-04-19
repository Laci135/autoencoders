import modules.Module
import numpy as np
import math

class Maxpool(int n):

    def __init(self, n):
        self.n = n

    def _build(self, X):
        self.i = X.shape[1]
        self.H = X.shape[2]
        self.W = X.shape[3]

        self.mask = np.zeros(ceil(self.H/self.n), ceil(self.W/self.n), dtype=(int, 2))

    def __readmax(data, at):
        for h in range(n):
            for w in range(n):
                coords = (at[0]*n + h, at[1]*n + w)
                max = None
                if coords[0] and coords[1] < W:
                    val = data[coords]
                    if max is None or val > max:
                        max = val
                        self.mask[at] = (h, w)
                   
    def _forward(self, X):
        assert X.shape[1:3] == (self.i, self.H, self.W), f"Maxpool module: Please fix input dimensions: {X.shape} -> {(X.shape[0], self.i, self.H, self.W)}"
       
        B = X.shape[0]

        out = np.zeros(ceil(H/n), ceil(W/n), dtype=float)
        
        for b in range(B):
            for i in range(self.I):
                for h in range(self.H):
                    for w in range(self.W):
                         max = __readmax(X, (b, i, h, w))
                         out[b, i, h, w] = max
    
    def backprop(self, lr, loss):
        assert loss.shape[1:3] == (self.i, ceil(self.H/self.n), ceil(self.H/self.n)), f"Maxpool module: Wrong backprop dimensions. Fix: {loss.shape} -> {(loss.shape[0], self.i, self.H*self.n, self.W*self.n)}"

        out = np.zeros((B, self.I, self.H, self.W))

        for b in range(B):
            for i in range(self.I):
                for h in range(self.H):
                    for w in range(self.W):
                        


        
