"""
A personal logistic regression model learning
"""

import numpy as np
from lab_utils import sigmoid

def compute_cost_logistic(X, y, w, b):
    """
    Args:
        X (ndarray(m,n))  : Data, m examples, n features
        y (ndarray(m,))   : target values
        w, b (scalar)     : model parameters
    Returns:
        cost (scalar)     : total cost
    """
    m, n = X.shape
    cost = 0.

    for i in range(m):
        z_i    = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost  += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    cost = cost / m

    return cost

