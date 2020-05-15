import modules
import numpy as np

# conv layer
class Conv(modules.Module):

    # parameters: kernel size, output channel count, padding -- tuple (left&top, right&bottom)
    # padding can be "same" instead to keep image size (assuming stride=1)
    def __init__(self, n, o, padding):
        super(Conv, self).__init__()
        self.n = n
        self.O = o
        # "same" padding support -- automatically calculates padding
        if padding == "same":
            self.padding = (n // 2, (n-1)//2)
        else:
            assert len(padding) == 2, "Conv module: padding must be set to 'same' or a tuple of two integers"
    
    # build on fwd pass
    def _build(self, X):
        self.H = X.shape[1] # input height, width and channel count
        self.W = X.shape[2]
        self.I = X.shape[3]
        # create conv filter (kernel_size*kernel_size*inp_channel*out_channel)
        self.__filter = np.ones((self.n, self.n, self.I, self.O))

    # conv operation -- params: inp data and position
    # this method performs the elementary convolution operation at a given position
    def __conv(self, data, at):
        total = 0
        for h in range(self.n): # convolution = discrete tim sum of the product of img pixels below the filter and corresponding filter data
            for w in range(self.n):
                total += data[at[0], at[1] + h, at[2] + w, at[3]] * self.__filter[h, w, at[3] , at[4]]
        return total / (self.n*self.n)
                
    # fwd pass
    # do not call explicitely
    def _forward(self, X):
        B = X.shape[0]
        assert X.shape == (B, self.H, self.W, self.I), f"Conv module: Please fix input dimensions: {X.shape} -> {(B, self.H, self.W, self.I)}"
        
        # calculate new size (for padded data)
        self.HH = self.H + sum(self.padding)
        self.WW = self.W + sum(self.padding)

        # create large array for padded data
        X_padded = np.zeros((B, self.HH, self.WW, self.I))

        # put data inside the large array
        X_padded[:, self.padding[0]:self.H+self.padding[0], self.padding[0]:self.W+self.padding[0]] = X

        # store data for backprop
        self.__last = X_padded 
        
        # calculate output size
        self.HH -= self.n-1
        self.WW -= self.n-1

        # create output data
        out = np.zeros((B, self.HH, self.WW, self.O))

        # perform convolution (for all inp/out channel pairings, all filter positions, all pixels under filter)
        for b in range(B):
            for hh in range(self.HH):
                for ww in range(self.WW):
                    for i in range(self.I):
                        for o in range(self.O): # use elemetary conv operation
                            out[b, hh, ww, o] = self.__conv(X_padded, (b, hh, ww, i, o))

        return out

    # elementary gradient operation
    def __spread_grad(self, to_inp, to_filter, grad, at):
  
        # get grad at a given location
        amount = grad[at[0], at[1], at[2], at[4]] / (self.n*self.n)
        
        # calculate input and filter gradients
        for h in range(self.n):
            for w in range(self.n):
                last_val = self.__last[at[0], at[1]+h, at[2]+w, at[3]]
                to_filter[h, w, at[3], at[4]] += amount * last_val
                filter_val =  self.__filter[h, w, at[3], at[4]]
                to_inp[at[0], at[1]+h, at[2]+w, at[3]] += amount * filter_val               

    # backprop -- must call explicitely when training
    def backprop(self, grad):
        B = grad.shape[0]
        assert grad.shape == (B, self.HH, self.WW, self.O), f"Conv module: Please fix input dimensions: {grad.shape} -> {(grad.shape[0], self.H, self.W, self.O)}"
         
        # initialize grad for filter and input data
        filter_grad = np.zeros((self.n, self.n, self.I, self.O))
        inp_grad = np.zeros((B, self.H+sum(self.padding), self.W+sum(self.padding), self.I))
        
        # calculate grad (for all inp/out channel pairings, all filter positions, all pixels under filter)
        for b in range(B):
            for h in range(self.HH):
                for w in range(self.WW):
                    for i in range(self.I):
                        for o in range(self.O):
                            self.__spread_grad(inp_grad, filter_grad, grad, (b, h, w, i, o))
        # discard data at the edge (inverse of applying padding)
        inp_grad = inp_grad[:, self.padding[0]:self.H+self.padding[0], self.padding[0]:self.W+self.padding[0]]
        
        # divide by inp size
        filter_grad /= self.H*self.W
        # apply grad to filter
        self.__filter += filter_grad
        
        # return backwards grad
        return inp_grad / (self.n*self.n)

        
