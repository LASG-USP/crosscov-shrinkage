# Cross-Covariance Shrinkage

Implementation of a shrinkage estimator for cross-covariance matrices.

## Installation

```
pip install crosscov-shrinkage
```
or
```
pip install git+https://github.com/LASG-USP/crosscov-shrinkage.git
```

## Quick Example

```python
import numpy as np
from src.crosscov import shrink_cross_covariance

X = np.random.randn(20,200)
Y = np.random.randn(10,200)

Sigma = shrink_cross_covariance(X,Y)
```


## Reference
If you use this package, please cite:
```bibtex
@article{ranazzi2026covariance,
  title={Covariance scaling: Theory, extension, and applications to ensemble-based history matching},
  author={Ranazzi, Paulo Henrique and Luo, Xiaodong and Sampaio, Marcio A.},
  journal={Computational Geosciences},
  year={2026},
  doi={10.1007/s10596-026-10413-w}
}
```