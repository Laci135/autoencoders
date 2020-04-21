import losses
import numpy as np
import math

class MSE(losses.Loss):
   
    def __init__(self):
        super(MSE, self).__init__()
    
    def _build(self, X):
        pass

    def _calculate(self, X, Y):
        assert Y is not None, "MSE: Pass the target as parameter to this method."
        assert X.shape == Y.shape, f"MSE: Please fix input to match target {X.shape} -> {Y.shape}"
        return (X*X-Y*Y) / 2
    
    def _calculate_grad(self, X, Y):
        assert Y is not None, "MSE: Pass the target as parameter to this method."
        assert X.shape == Y.shape, f"MSE: Please fix input to match target {X.shape} -> {Y.shape}"
        return -X

    def _calculate_total(self, X, Y):
        assert Y is not None, "MSE: Pass the target as parameter to this method."
        return np.sum((X*X-Y*Y) / 2)
