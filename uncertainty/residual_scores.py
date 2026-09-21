"""Residual-based score fields for localization."""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

from solver.residual import residual


def rhs_vector(A: sp.spmatrix, f: np.ndarray | float) -> np.ndarray:
    """Return f as a vector compatible with A, accepting scalar sources."""
    n = A.shape[0]
    arr = np.asarray(f, dtype=np.float64)
    if arr.ndim == 0 or arr.size == 1:
        return np.full(n, float(arr.reshape(-1)[0]), dtype=np.float64)
    vec = arr.reshape(-1)
    if vec.size != n:
        raise ValueError(f"f has {vec.size} entries; A has {n} rows")
    return vec


def infer_field_shape(A: sp.spmatrix, *arrays: np.ndarray | float) -> tuple[int, int]:
    """Infer a square field shape from array inputs or A."""
    n = A.shape[0]
    for arr in arrays:
        a = np.asarray(arr)
        if a.ndim == 2 and a.size == n:
            return int(a.shape[0]), int(a.shape[1])
    N = int(round(math.sqrt(n)))
    if N * N != n:
        raise ValueError(f"A size must be a square grid; got {n} unknowns")
    return N, N


def residual_vector(A: sp.spmatrix, u_hat: np.ndarray, f: np.ndarray | float) -> np.ndarray:
    """Compute r = A u_hat - f as a vector."""
    return residual(A, u_hat, rhs_vector(A, f))


def residual_score(A: sp.spmatrix, u_hat: np.ndarray, f: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(r, |r|)`` as fields."""
    shape = infer_field_shape(A, u_hat, f)
    r = residual_vector(A, u_hat, f).reshape(shape)
    return r, np.abs(r)
