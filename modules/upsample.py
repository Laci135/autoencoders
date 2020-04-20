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
        for h in range(n):
            for w in range(n):
                data[at[0], at[1], at[2]+h, at[3]+w] = val

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
    
    def __readavg(data, at):
        total = 0
        for h in range(n):
            for w in range(n):
                total += data[at[0], at[1], at[2]+h, at[3]+w]
        return total / (self.n*self.n)


    def backprop(self, lr, loss):
        assert loss is not None, "Upsample module: Loss is None."

        out = np.zeros((B, self.I, self.H, self.W))

        for b in range(B):
            for i in range(self.I):
                for h in range(self.H):
                    for w in range(self.W):
                        val = self.__readavg(loss, (b, i, h, w))
                        out[b, i, h, w] = val
