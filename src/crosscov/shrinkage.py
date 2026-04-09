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


def compute_localization_approximate(x, y=None, ddof=1):
    n = x.shape[-1] - ddof
    if y is None:
        corr = np.corrcoef(x)
        varx = np.var(x, axis=-1, ddof=ddof)
        vary = varx
    else:
        cov = cross_covariance(x, y)
        varx = np.var(x, axis=-1, ddof=ddof)
        vary = np.var(y, axis=-1, ddof=ddof)

        stdx = np.sqrt(varx)
        stdy = np.sqrt(vary)
        corr = cov / np.outer(stdx, stdy)

    eps = 1e-12
    corr_safe = np.where(np.abs(corr) < eps, eps, corr)

    phi = 1.0 / (corr_safe ** 2)
    P = np.zeros_like(corr)
    mask = n >= phi
    P[mask] = (n - phi[mask]) / (n + 1)
    return P
