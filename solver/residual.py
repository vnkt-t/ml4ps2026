"""Discrete residual r = A u_hat - b, computed with the SAME A as the solve.

The single rule the plan calls non-negotiable: the residual must use the assembled
discrete operator A(a) from assemble_fd.py, never an independent continuous-residual
stencil applied to a grid output. Verified by tests/test_residual_identity.py.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def residual(A: sp.spmatrix, u_hat: np.ndarray, b: np.ndarray) -> np.ndarray:
    """r = A u_hat - b. Inputs may be fields (N, N) or vectors; returns a vector."""
    uh = np.asarray(u_hat, dtype=np.float64).reshape(-1)
    rhs = np.asarray(b, dtype=np.float64).reshape(-1)
    if A.shape[0] != A.shape[1] or uh.shape[0] != A.shape[0] or rhs.shape[0] != A.shape[0] or rhs.size == 0:
        raise ValueError(f"shape mismatch: A {A.shape}, u_hat {uh.shape}, b {rhs.shape}")
    matrix = A.tocsr()
    if not all(np.all(np.isfinite(x)) for x in (uh, rhs, matrix.data)):
        raise ValueError("A, u_hat, and b must be finite")
    result = matrix.dot(uh) - rhs
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("the discrete residual is non-finite")
    return result
