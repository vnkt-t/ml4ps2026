"""Ground-truth solve and full-inverse oracle for A(a) u = b.

Direct sparse solve (SciPy) — exact to round-off for the small grids in this project,
which is what the residual-identity gate (Experiment 1) needs. CG/multigrid live in
relaxation.py / multigrid.py as the cheaper compute-ladder rungs.
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def solve(A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
    """Solve A x = b. `b` may be a field (N, N) or a vector (N^2,). Returns a vector."""
    rhs = np.asarray(b, dtype=np.float64).reshape(-1)
    if A.shape[0] != A.shape[1] or rhs.shape[0] != A.shape[0] or rhs.size == 0:
        raise ValueError(f"b has {rhs.shape[0]} entries; A is {A.shape}")
    matrix = A.tocsc()
    if not np.all(np.isfinite(rhs)) or not np.all(np.isfinite(matrix.data)):
        raise ValueError("A and b must be finite")
    with warnings.catch_warnings():
        warnings.simplefilter("error", spla.MatrixRankWarning)
        try:
            x = spla.spsolve(matrix, rhs)
        except spla.MatrixRankWarning as exc:
            raise FloatingPointError("the discrete system is singular") from exc
    if not np.all(np.isfinite(x)):
        raise FloatingPointError("the discrete solve produced a non-finite solution")
    return np.asarray(x, dtype=np.float64)


def solve_field(A: sp.spmatrix, b: np.ndarray, N: int) -> np.ndarray:
    """Same as solve() but returns the (N, N) field."""
    return solve(A, b).reshape(N, N)
