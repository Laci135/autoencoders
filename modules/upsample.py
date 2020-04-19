import modules

class Upsample(modules.Module):

    def __init__(self, n):
        self.n = n

    def _build(self, X):
        self.i = X.shape[1]
        self.H = X.shape[2]
        self.W = X.shape[3]

    def __fill(data, at, val):
        for h in range(n):
            for w in range(n):
                data[at[0]+h, at[1]+w] = val

    def _forward(self, X):
        out = np.zeros((i, self.H*n, self.W*n))

        for h in range(X.shape[2]):
            for w in range(X.shape[3]):
                val = X[h, w]
                __fill(out, (h, w), val)
