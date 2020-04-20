import modules

class Upsample(modules.Module):

    def __init__(self, n):
        super(Upsample, self).__init__()
        self.n = n

    def _build(self, X):
        self.I = X.shape[1]
        self.H = X.shape[2]
        self.W = X.shape[3]

    def __fill(data, at, val):
        for h in range(n):
            for w in range(n):
                data[at[0], at[1], at[2]+h, at[3]+w] = val

    def _forward(self, X):
        assert X.size[1:3] == (self.H, self.W)

        B = X.size[0]
        out = np.zeros((B, self.i, self.H*n, self.W*n))
       
        for b in range(B):
            for i in range(self.I):
                for h in range(self.H):
                    for w in range(self.W):
                        val = X[b, i, h, w]
                        __fill(out, (b, i, h, w), val)
    
    def __readavg(data, at):
        total = 0
        for h in range(n):
            for w in range(n):
                total += data[at[0], at[1], at[2]+h, at[3]+w]
        return total / (n*n)


    def backprop(self, lr, loss):
        assert loss is not None, "Upsample module: Loss is None."

        out = np.zeros((B, self.I, self.H, self.W))

        for b in range(B):
            for i in range(self.I):
                for h in range(self.H):
                    for w in range(self.W):
                        val = __readavg(loss, (b, i, h, w))
                        out[b, i, h, w] = val
