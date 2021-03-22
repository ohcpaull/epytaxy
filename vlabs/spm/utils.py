import numpy as np
# Curve fitting functions

def gaussian(x, A, x0):
    return A * np.exp(x0 - x)