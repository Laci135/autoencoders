import modules
import numpy as np

# upsample module
class Upsample(modules.Module):

    # params -- n: pool size
    def __init__(self, n):
        super(Upsample, self).__init__()
        self.n = n
    
    # do not call explicitely
    def _build(self, X):
        self.H = X.shape[1]
        self.W = X.shape[2]
        self.I = X.shape[3]

    # elementary upsample operation
    # fills a pool with a given value
    def __fill(self, data, at, val):
        for h in range(self.n):
            for w in range(self.n):
                data[at[0], at[1]*self.n+h, at[2]*self.n+w, at[3]] = val
    
    # fwd pass
    # do not call explicitely
    def _forward(self, X):
        B = X.shape[0]
        assert X.shape == (B, self.H, self.W, self.I), f"Upsample module: Please fix input dimensions: {X.shape} -> {(B, self.H, self.W, self.I)}"
        
        # calculate output size
        self.HH = self.H*self.n
        self.WW = self.W*self.n

        # create output tensor
        out = np.zeros((B, self.HH, self.WW, self.I))
        
        # perform fwd pass (upsample) for all pools
        for b in range(B):
            for h in range(self.H):
                for w in range(self.W):
                    for i in range(self.I):
                        val = X[b, h, w, i]
                        self.__fill(out, (b, h, w, i), val)

        return out
    
    # elementary upsample grad operation
    def __readtotal(self, data, at):
        total = 0
        for h in range(self.n): # sum up the gradients in the pool
            for w in range(self.n):
                total += data[at[0], at[1]*self.n+h, at[2]*self.n+w, at[3]]
        return total

    # backprop
    # must call explicitely
    def backprop(self, grad):
        assert grad is not None, "Upsample module: Grad is None."
        B = grad.shape[0]

        # create inp grad mx
        inp_grad = np.zeros((B, self.H, self.W, self.I))

        # calculate inp grad for all pools
        for b in range(B):
            for i in range(self.I):
                for h in range(self.H):
                    for w in range(self.W):
                        val = self.__readtotal(grad, (b, h, w, i))
                        inp_grad[b, h, w, i] = val # use the elementary upsample grad operation

        return inp_grad / (self.n*self.n) # divide by the pool size
