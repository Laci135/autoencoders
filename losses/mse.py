import losses
import numpy as np
import math

class MSE(losses.Loss):
    
    def _calculate(self, X, Y):    
        assert X.shape == Y.shape, f"MSE: Please fix input to match target {X.shape} -> {Y.shape}"
        _set((X*X-Y*Y) / 2)
    
    def _calculate_grad(self, X, Y):
        assert X.shape == Y.shape, f"MSE: Please fix input to match target {X.shape} -> {Y.shape}"
        _set(gradient(-X)

    def total(self):
        return np.mean(__loss)
