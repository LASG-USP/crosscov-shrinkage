import numpy as np
from .covariance import cross_covariance


def compute_scaling_approximate(x, y, ddof=0, norm=None):
    assert x.shape[-1] == y.shape[-1], (
        f"Last dimension mismatch: x.shape[-1]={x.shape[-1]} != y.shape[-1]={y.shape[-1]}"
    )
    n = x.shape[-1]
    if norm is not None:
        y = np.diag(1 / np.sqrt(norm.diagonal())) @ y
    C = cross_covariance(x, y)
    tr2c = np.sum(np.var(x, axis=-1, ddof=ddof))*np.sum(np.var(y, axis=-1, ddof=ddof))

    trc2 = np.trace(C @ C.T)
    return max(0, (trc2 - tr2c/(n-ddof)) / (trc2+trc2/(n-ddof)))


def shrink_cross_covariance(X, Y, ddof=0):

    alpha = compute_scaling_approximate(X, Y, ddof)

    C = cross_covariance(X, Y, ddof)

    return alpha* C


def compute_localization_approximate(x, n, y=None, ddof=1):
    n -= ddof
    if type == 'full':
        C = np.cov(x, ddof=ddof)
        varx = np.diag(C)
        vary = np.diag(C)
    if y is not None:
        C = cross_covariance(x, y)
        varx = np.var(x, axis=-1, ddof=ddof)
        vary = np.var(y, axis=-1, ddof=ddof)
    P = np.zeros(C.shape)

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            phi = (varx[i]*vary[j])/(C[i, j] ** 2)
            P[i, j] = 0 if n < phi else (n-phi)/(n+1)

    return P
