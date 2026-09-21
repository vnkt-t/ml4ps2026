"""Flux and energy diagnostics consistent with the assembled TPFA operator.

The residual is the *divergence* of face fluxes, rather than their magnitude.
Cell energies here partition the discrete quadratic form and do not reuse the
pointwise product ``abs(e * residual)`` as an energy proxy.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def cell_energy(A: sp.spmatrix, error: np.ndarray) -> np.ndarray:
    """Nonnegative cell contributions summing to ``error.T @ A @ error``.

    For a symmetric diffusion M-matrix, write ``w_ij = -A_ij >= 0`` and
    ``b_i = sum_j A_ij >= 0`` (Dirichlet boundary leakage). Each interior edge
    contributes half of ``w_ij * (e_i-e_j)**2`` to either cell; boundary faces
    contribute ``b_i * e_i**2`` to their adjacent cell. The return shape matches
    ``error``. For the 1/h²-scaled operator in this project, multiply by h²
    before summing to approximate the domain-integrated energy.
    """
    e_arr = np.asarray(error, dtype=np.float64)
    e = e_arr.reshape(-1)
    a = A.tocsr().astype(np.float64)
    if a.shape != (e.size, e.size) or e.size == 0:
        raise ValueError(f"shape mismatch: A {a.shape}, error {e_arr.shape}")
    if not np.all(np.isfinite(e)) or not np.all(np.isfinite(a.data)):
        raise ValueError("A and error must be finite")
    # Allow round-off in symmetry / row sums without silently accepting an
    # operator for which this nonnegative energy partition is invalid.
    scale = float(np.max(np.abs(a.data))) if a.nnz else 0.0
    tol = 64.0 * np.finfo(np.float64).eps * scale
    asym = a - a.T
    if asym.nnz and float(np.max(np.abs(asym.data))) > tol:
        raise ValueError("cell_energy requires a symmetric diffusion operator")
    off = a.tocoo()
    mask = off.row != off.col
    if np.any(off.data[mask] > tol):
        raise ValueError("cell_energy requires nonpositive off-diagonal entries")
    leakage = np.asarray(a.sum(axis=1)).reshape(-1)
    if np.any(leakage < -tol):
        raise ValueError("cell_energy requires nonnegative boundary leakage")
    energy = np.maximum(leakage, 0.0) * e**2
    edge = 0.5 * np.maximum(-off.data[mask], 0.0) * (
        e[off.row[mask]] - e[off.col[mask]]
    ) ** 2
    np.add.at(energy, off.row[mask], edge)
    return energy.reshape(e_arr.shape)


def face_fluxes(a: np.ndarray, error: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return oriented TPFA ``-a grad(error)`` fluxes, including zero-BC walls.

    ``qx`` has shape ``(N+1, N)`` and ``qy`` shape ``(N, N+1)``; positive values
    point in the increasing coordinate direction. Their cell divergence is
    exactly the assembled operator applied to ``error``, up to round-off.
    """
    coeff = np.asarray(a, dtype=np.float64)
    e = np.asarray(error, dtype=np.float64)
    if coeff.ndim != 2 or coeff.shape[0] != coeff.shape[1] or coeff.size == 0:
        raise ValueError(f"a must be a nonempty square field; got {coeff.shape}")
    if e.shape != coeff.shape:
        raise ValueError(f"a and error shapes differ: {coeff.shape} vs {e.shape}")
    if not np.all(np.isfinite(coeff)) or np.any(coeff <= 0) or not np.all(np.isfinite(e)):
        raise ValueError("a must be finite and positive; error must be finite")
    n = coeff.shape[0]
    qx = np.empty((n + 1, n), dtype=np.float64)
    qy = np.empty((n, n + 1), dtype=np.float64)

    def harmonic(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        lo, hi = np.minimum(left, right), np.maximum(left, right)
        return lo * (2.0 / (1.0 + lo / hi))

    qx[1:-1] = -n * harmonic(coeff[:-1], coeff[1:]) * np.diff(e, axis=0)
    qy[:, 1:-1] = -n * harmonic(coeff[:, :-1], coeff[:, 1:]) * np.diff(e, axis=1)
    qx[0], qx[-1] = -2.0 * n * coeff[0] * e[0], 2.0 * n * coeff[-1] * e[-1]
    qy[:, 0], qy[:, -1] = -2.0 * n * coeff[:, 0] * e[:, 0], 2.0 * n * coeff[:, -1] * e[:, -1]
    return qx, qy


def cell_flux_magnitude(a: np.ndarray, error: np.ndarray) -> np.ndarray:
    """Cell flux target: root-sum-square of the two face-averaged components.

    Average *squared* fluxes on opposite faces before summing the components,
    so oppositely signed face fluxes cannot cancel within this magnitude.
    This is a diagnostic allocation of face fluxes, not a new cell gradient.
    """
    qx, qy = face_fluxes(a, error)
    return np.sqrt(0.5 * (qx[:-1]**2 + qx[1:]**2 + qy[:, :-1]**2 + qy[:, 1:]**2))
