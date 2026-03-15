import numpy as np

def cross_covariance(X, Y, ddof=0):
    """
    Empirical cross-covariance matrix.
    """

    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)

    n = X.shape[1]

    return (Xc @ Yc.T) / (n - ddof)