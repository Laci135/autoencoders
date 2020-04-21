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
    
        self.__filter = np.ones((self.n, self.n, self.I, self.O))

    def __conv(self, data, at):
        total = 0
        for h in range(self.n):
            for w in range(self.n):
                total += data[at[0], at[1] + h, at[2] + w, at[3]] * self.__filter[h, w, at[3] , at[4]]
        return total / (self.n*self.n)
                

    def _forward(self, X):
        B = X.shape[0]
        assert X.shape == (B, self.H, self.W, self.I), f"Conv module: Please fix input dimensions: {X.shape} -> {(B, self.H, self.W, self.I)}"
        
        self.HH = self.H + sum(self.padding)
        self.WW = self.W + sum(self.padding)

        X_padded = np.zeros((B, self.HH, self.WW, self.I))

        X_padded[:, self.padding[0]:self.H+self.padding[0], self.padding[0]:self.W+self.padding[0]] = X

        self.__last = X_padded 

        self.HH -= self.n-1
        self.WW -= self.n-1

        out = np.zeros((B, self.HH, self.WW, self.O))

        for b in range(B):
            for hh in range(self.HH):
                for ww in range(self.WW):
                    for i in range(self.I):
                        for o in range(self.O):
                            out[b, hh, ww, o] = self.__conv(X_padded, (b, hh, ww, i, o))

        return out

    def __spread_grad(self, to_inp, to_filter, grad, at):
  
        amount = grad[at[0], at[1], at[2], at[4]] / (self.n*self.n)
        for h in range(self.n):
            for w in range(self.n):
                last_val = self.__last[at[0], at[1]+h, at[2]+w, at[3]]
                to_filter[h, w, at[3], at[4]] += amount * last_val
                filter_val =  self.__filter[h, w, at[3], at[4]]
                to_inp[at[0], at[1]+h, at[2]+w, at[3]] += amount * filter_val               

    def backprop(self, grad):
        B = grad.shape[0]
        assert grad.shape == (B, self.HH, self.WW, self.O), f"Conv module: Please fix input dimensions: {grad.shape} -> {(grad.shape[0], self.H, self.W, self.O)}"

        filter_grad = np.zeros((self.n, self.n, self.I, self.O))
        inp_grad = np.zeros((B, self.H+sum(self.padding), self.W+sum(self.padding), self.I))
        
        for b in range(B):
            for h in range(self.HH):
                for w in range(self.WW):
                    for i in range(self.I):
                        for o in range(self.O):
                            self.__spread_grad(inp_grad, filter_grad, grad, (b, h, w, i, o))

        inp_grad = inp_grad[:, self.padding[0]:self.H+self.padding[0], self.padding[0]:self.W+self.padding[0]]

        filter_grad /= self.H*self.W
        self.__filter += filter_grad

        return inp_grad / (self.n*self.n)

        
