import modules
import numpy as np

class Conv(modules.Module):

    def __init__(self, n, o, stride, padding):
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
    
        self.kernel = np.array((self.n, self.n))

    def _forward(self, X):
        assert X.shape[1:3] == (self.i, self.H, self.W), f"Conv module: Please fix input dimensions: {X.shape} -> {(X.shape[0], self.i, self.H, self.W)}"
        
        B = X.shape[0]
        
        out = np.array((

        for b in range(B):
            for i in range(self.I):
                for h in np.r_[0 : (H+sum(self.padding))/stride + 1 : stride]:
                    for w in np.r_[0 : (W+sum(self.padding))/stride + 1 : stride]:
                        



