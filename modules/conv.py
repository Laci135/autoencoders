import modules
import numpy as np

class Conv(modules.Module):

    def __init__(self, n, o, padding):
        self.n = n
        self.o = o
        self.stride = stride

        if padding == "same":
            self.padding = (n // 2, (n-1)//2)
        else:
            assert len(padding) == 2, "Conv module: padding must be set to 'same' or a tuple of two integers"
      


    def build(self, X):
        self.i = X.shape[1]
        self.H = X.shape[2]
        self.W = X.shape[3]
    
        self.kernel = np.zeros((self.n, self.n))

    def __conv(self, data, at):
        total = 0
        for h in range(self.n):
            for w in range(self.n):
                total += data[at[2] + h, at[3] + w] * filter[h, w]
        return total / (self.n*self.n)
                

    def _forward(self, X):
        assert X.shape[1:3] == (self.i, self.H, self.W), f"Conv module: Please fix input dimensions: {X.shape} -> {(X.shape[0], self.i, self.H, self.W)}"
        
        B = X.shape[0]
        
        self.HH = H + sum(self.padding)
        self.WW = W + sum(self.padding)

        X_padded = np.array((B, I, HH, WW))

        X_padded[self.padding[0]:H+self.padding[0], self.padding[1]:W+self.padding[1] = X

        self.HH -= self.n-1
        self.WW -= self.n-1

        out = np.array((B, self.I, self.HH, self.WW))

        for b in range(B):
            for i in range(self.I):
                for hh in range(self.HH):
                    for ww in range(self.WW):
                        out[b, i, hh, ww] = __conv(X_padded, (b, i, hh, ww))
        return out

    def __grad(self, data, at):
        return pass

    def backprop(self, lr, grad):
        assert loss.shape[1:3] == (self.i, self.HH, self.WW), f"Conv module: Please fix input dimensions: {loss.shape} -> {(loss.shape[0], self.i, self.H, self.W)}"

        B = loss.shape[0]

        out = np.array((B, self.I, self.H, self.W))

        for b in range(B):
            for i in range(self.I):
                for h in range(self.H):
                    for w in range(self.W):
                        out[b, i, h, w] = __grad(grad, (b, i, h, w))

        return out

        
