import modules
import numpy as np

class Conv(modules.Module):

    def __init__(self, n, o, padding):
        super(Conv, self).__init__()
        self.n = n
        self.O = o

        if padding == "same":
            self.padding = (n // 2, (n-1)//2)
        else:
            assert len(padding) == 2, "Conv module: padding must be set to 'same' or a tuple of two integers"

    def _build(self, X):
        self.H = X.shape[1]
        self.W = X.shape[2]
        self.I = X.shape[3]
    
        self.filter = np.ones((self.n, self.n, self.I, self.O))

    def __conv(self, data, at):
        total = 0
        for h in range(self.n):
            for w in range(self.n):
                    total += data[at[0], at[1] + h, at[2] + w, at[3]] * self.filter[h, w, at[3] , at[4]]
        return total / (self.n*self.n)
                

    def _forward(self, X):
        B = X.shape[0]
        assert X.shape == (B, self.H, self.W, self.I), f"Conv module: Please fix input dimensions: {X.shape} -> {(B, self.H, self.W, self.I)}"
           
        self.HH = self.H + sum(self.padding)
        self.WW = self.W + sum(self.padding)

        X_padded = np.zeros((B, self.HH, self.WW, self.I))

        X_padded[:, self.padding[0]:self.H+self.padding[0], self.padding[1]:self.W+self.padding[1]] = X

        self.HH -= self.n-1
        self.WW -= self.n-1

        out = np.zeros((B, self.HH, self.WW, self.I))

        for b in range(B):
            for hh in range(self.HH):
                for ww in range(self.WW):
                    for i in range(self.I):
                        for o in range(self.O):
                            out[b, hh, ww, i] = self.__conv(X_padded, (b, hh, ww, i, o))

        return out

    def __grad(self, data, at):
        total = 0
        return total

    def backprop(self, lr, grad):
        assert loss.shape[1:3] == (self.HH, self.WW, self.I), f"Conv module: Please fix input dimensions: {loss.shape} -> {(loss.shape[0], self.H, self.W, self.I)}"

        B = loss.shape[0]

        out = np.array((B, self.H, self.W, self.I))

        for b in range(B):
            for h in range(self.H):
                for w in range(self.W):
                    for i in range(self.I):
                        out[b, h, w, i] = __grad(grad, (b, h, w, i))
        return out

        
