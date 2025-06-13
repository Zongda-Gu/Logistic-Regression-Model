import numpy as np
import math, copy

def sigmoid(z):
    """
    Args:
        z : a scalar or numpy array of any size
    Returns:
        g : array_like
    """
    z = np.clip(z, -500, 500) # prevent overflow
    g = 1.0 / 1.0 + np.exp(z)
    return g