"""Fast constant-coefficient inverse for the project's cell-centered TPFA grid.

The one-dimensional Dirichlet matrix has diagonal (3,2,...,2,3), rather than
the nodal-grid diagonal (2,...,2). Its eigenvectors are
``sin(pi*k*(i+1/2)/N)``, k=1,...,N, so an orthonormal DST-II diagonalizes it.
The final sine mode has a different normalization; scipy's orthonormal forward
and inverse transforms handle that detail. No hierarchy or factorization of the
variable-coefficient operator is needed.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.fft import dstn, idstn
from scipy.sparse.linalg import LinearOperator


@lru_cache(maxsize=16)
def _eigenvalues(n: int) -> np.ndarray:
    modes = np.arange(1, n + 1, dtype=np.float64)
    one_d = 4.0 * n**2 * np.sin(np.pi * modes / (2.0 * n))**2
    eigenvalues = one_d[:, None] + one_d[None, :]
    eigenvalues.setflags(write=False)
    return eigenvalues


def solve_poisson(rhs: np.ndarray, coefficient: float = 1.0) -> np.ndarray:
    """Solve constant-a TPFA ``-a Laplacian(x)=rhs`` with zero Dirichlet walls.

    ``rhs`` is a nonempty square field. Returns a field of the same shape.
    Work is O(N² log N), with O(N²) storage, including the first call.
    """
    field = np.asarray(rhs, dtype=np.float64)
    if field.ndim != 2 or field.shape[0] != field.shape[1] or field.size == 0:
        raise ValueError(f"rhs must be a nonempty square field; got {field.shape}")
    if not np.all(np.isfinite(field)):
        raise ValueError("rhs must be finite")
    if not np.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("coefficient must be finite and strictly positive")
    transformed = dstn(field, type=2, norm="ortho")
    transformed /= coefficient * _eigenvalues(field.shape[0])
    return idstn(transformed, type=2, norm="ortho")


def poisson_preconditioner(n: int, coefficient: float = 1.0) -> LinearOperator:
    """SPD constant-Poisson inverse as a SciPy CG preconditioner on N² vectors."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    if not np.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("coefficient must be finite and strictly positive")

    def apply(vector: np.ndarray) -> np.ndarray:
        return solve_poisson(np.asarray(vector).reshape(n, n), coefficient).reshape(-1)

    return LinearOperator((n*n, n*n), matvec=apply, rmatvec=apply, dtype=np.float64)
