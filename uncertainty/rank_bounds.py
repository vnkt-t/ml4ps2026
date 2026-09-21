"""Analytic pointwise bounds and top-set separation for TPFA corrections.

Let B=A(1), A >= a_min B in Loewner order, and d=r-Az. In exact arithmetic,
|e_i-z_i| <= sqrt((B^-1)_ii * d.T B^-1 d) / a_min. This is a consequence of
Cauchy-Schwarz in the A inner product and order reversal under inversion.
It is a deterministic discrete bound, not a conformal/probabilistic interval.
Float64 evaluation is not a formally roundoff-enclosed certificate.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from solver.poisson import solve_poisson


@lru_cache(maxsize=16)
def poisson_inverse_diagonal(n: int) -> np.ndarray:
    """Diagonal of the inverse cell-centered 2D Dirichlet Poisson operator.

    The dense N-by-N factors are one-dimensional sine bases, not N²-by-N²
    inverse matrices. O(N³) one-time work and O(N²) storage per grid.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    modes = np.arange(1, n + 1, dtype=np.float64)
    position = np.arange(n, dtype=np.float64) + 0.5
    q = np.sqrt(2.0 / n) * np.sin(np.pi * position[:, None] * modes[None, :] / n)
    q[:, -1] /= np.sqrt(2.0)
    weights = q * q
    eigenvalues = 4.0 * n**2 * np.sin(np.pi * modes / (2 * n))**2
    inverse_eigenvalues = 1.0 / (eigenvalues[:, None] + eigenvalues[None, :])
    # Explicit contractions avoid platform BLAS floating-status warnings on
    # this positive, well-scaled sum; precomputation is only O(N³) once/grid.
    partial = np.einsum("ik,kl->il", weights, inverse_eigenvalues, optimize=False)
    diagonal = np.einsum("il,jl->ij", partial, weights, optimize=False)
    if not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
        raise FloatingPointError("invalid inverse diagonal")
    diagonal.setflags(write=False)
    return diagonal


def poisson_error_bound(defect: np.ndarray, a_min: float) -> np.ndarray:
    """Bound signed correction error given d=r-Az and true coefficient minimum.

    Requires the variable-coefficient operator to use the SAME grid, boundary
    convention, and TPFA weights as solver.assemble_fd. The caller must supply
    a lower bound on every coefficient; overestimating a_min invalidates the
    analytic guarantee. The defect and returned bounds have shape (N,N).
    """
    d = np.asarray(defect, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or d.size == 0 or not np.isfinite(d).all():
        raise ValueError("defect must be a finite nonempty square field")
    if not np.isfinite(a_min) or a_min <= 0:
        raise ValueError("a_min must be finite and positive")
    potential = solve_poisson(d)
    dual_energy = float(np.sum(d * potential))
    if not np.isfinite(dual_energy) or dual_energy < 0:
        raise FloatingPointError("invalid Poisson dual energy")
    return np.sqrt(poisson_inverse_diagonal(d.shape[0]) * dual_energy) / a_min


def magnitude_envelope(correction: np.ndarray, bound: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reverse-triangle enclosure of |e| from an enclosure of e-z."""
    z, b = np.asarray(correction, dtype=np.float64), np.asarray(bound, dtype=np.float64)
    if z.shape != b.shape or z.size == 0 or not np.isfinite(z).all() or not np.isfinite(b).all() or np.any(b < 0):
        raise ValueError("correction and bound must match, be finite, and bound nonnegative")
    return np.maximum(np.abs(z) - b, 0), np.abs(z) + b


def separate_top_set(lower: np.ndarray, upper: np.ndarray, q: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Cells certainly inside/outside the largest ceil(q*n) magnitudes.

    Strict separation avoids certifying an ambiguous tie. Cells with neither
    flag remain unresolved. The guarantee assumes valid input enclosures.
    """
    lo, hi = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    if lo.shape != hi.shape or lo.size == 0 or not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("enclosures must be finite, nonempty, and have matching shape")
    if np.any(lo < 0) or np.any(hi < lo) or not 0 < q <= 1:
        raise ValueError("require 0 <= lower <= upper and 0 < q <= 1")
    k = int(np.ceil(q * lo.size))
    if k == lo.size:
        return np.ones(lo.shape, dtype=bool), np.zeros(lo.shape, dtype=bool)
    upper_k_plus_one = np.partition(hi.reshape(-1), lo.size - k - 1)[lo.size - k - 1]
    lower_k = np.partition(lo.reshape(-1), lo.size - k)[lo.size - k]
    return lo > upper_k_plus_one, hi < lower_k
