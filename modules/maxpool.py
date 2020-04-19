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

        self.mask = np.zeros(ceil(H/n), ceil(W/n), dtype=(int, 2))

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
        
        out = np.zeros(ceil(H/n), ceil(W/n), dtype=float)
        
        for h in range(H.shape[0]):
            for w in range(W.shape[1]):
                 max = __readmax(X, (h, w))
                 out[h, w] = max

