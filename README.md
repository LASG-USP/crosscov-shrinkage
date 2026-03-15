# Cross-Covariance Shrinkage

Implementation of a shrinkage estimator for cross-covariance matrices.

## Installation

pip install crosscov

## Example

import numpy as np
from crosscov import shrink_cross_covariance

X = np.random.randn(20,200)
Y = np.random.randn(10,200)

Sigma = shrink_cross_covariance(X,Y)