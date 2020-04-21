import modules
import numpy as np

class Upsample(modules.Module):

    def __init__(self, n):
        super(Upsample, self).__init__()
        self.n = n

    def _build(self, X):
        self.H = X.shape[1]
        self.W = X.shape[2]
        self.I = X.shape[3]

    def __fill(self, data, at, val):
        for h in range(self.n):
            for w in range(self.n):
                data[at[0], at[1]+h, at[2]+w, at[3]] = val

    def _forward(self, X):
        B = X.shape[0]
        assert X.shape == (B, self.H, self.W, self.I), f"Upsample module: Please fix input dimensions: {X.shape} -> {(B, self.H, self.W, self.I)}"

        self.HH = self.H*self.n
        self.WW = self.W*self.n

        out = np.zeros((B, self.HH, self.WW, self.I))
       
        for b in range(B):
            for h in range(self.H):
                for w in range(self.W):
                    for i in range(self.I):
                        val = X[b, h, w, i]
                        self.__fill(out, (b, h, w, i), val)

        return out
    
    def __readtotal(self, data, at):
        total = 0
        for h in range(self.n):
            for w in range(self.n):
                total += data[at[0], at[1]+h, at[2]+w, at[3]]
        return total

    def backprop(self, lr, grad):
        assert grad is not None, "Upsample module: Grad is None."
        B = grad.shape[0]

        out = np.zeros((B, self.H, self.W, self.I))

        for b in range(B):
            for i in range(self.I):
                for h in range(self.H):
                    for w in range(self.W):
                        val = self.__readtotal(grad, (b, h, w, i))
                        out[b, h, w, i] = val

        return out
