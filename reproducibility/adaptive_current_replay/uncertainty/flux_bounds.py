"""Classical equilibrated-flux tightening of the discrete Poisson majorant.

Let A=D W_A D.T and B=D W_B D.T be the variable- and unit-coefficient TPFA
matrices, including Dirichlet-wall faces. For p=B^-1 d, the face flow
q=W_B D.T p satisfies D q=d. The minimum-energy (Thomson) principle gives

    d.T A^-1 d <= q.T W_A^-1 q = E_eq.

Consequently |(A^-1 d)_i| <= sqrt((B^-1)_ii * E_eq / a_min).
Since W_A >= a_min W_B, this is no larger in exact arithmetic than
sqrt((B^-1)_ii * d.T B^-1 d) / a_min. Both use the same single Poisson solve;
the tightening adds coefficient-weighted face sums. This is a classical flux
majorant, not a new theorem or a formally roundoff-enclosed certificate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solver.poisson import solve_poisson
from uncertainty.rank_bounds import poisson_inverse_diagonal


@dataclass(frozen=True)
class FluxBoundResult:
    """Two majorants sharing one Poisson solve and their energy diagnostics."""

    bound: np.ndarray
    baseline_bound: np.ndarray
    poisson_potential: np.ndarray
    equilibrated_energy: float
    poisson_energy: float
    a_min: float


def _matched_fields(coefficient: np.ndarray, potential: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(coefficient, dtype=np.float64)
    p = np.asarray(potential, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.size == 0 or p.shape != a.shape:
        raise ValueError("coefficient and potential must be matching nonempty square fields")
    if not np.isfinite(a).all() or np.any(a <= 0.0) or not np.isfinite(p).all():
        raise ValueError("coefficient must be finite/positive and potential finite")
    return a, p


def equilibrated_flux_energy(coefficient: np.ndarray, poisson_potential: np.ndarray) -> float:
    """Return sum(q_face**2 / W_A_face) for q=W_B D.T p.

    The divergence equals a defect d only when the supplied p solves B p=d.
    Interior inverse harmonic coefficients are (1/a_i + 1/a_j)/2. Each wall
    contributes 2*N**2*p_i**2/a_i, including each incident wall at corner cells.
    This face scaling matches solver.assemble_fd's cell-centered TPFA operator.
    """
    a, p = _matched_fields(coefficient, poisson_potential)
    n = a.shape[0]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        inverse_a = 1.0 / a
        interior_x = np.sum(np.diff(p, axis=0) ** 2 * (0.5 * inverse_a[:-1] + 0.5 * inverse_a[1:]))
        interior_y = np.sum(np.diff(p, axis=1) ** 2 * (0.5 * inverse_a[:, :-1] + 0.5 * inverse_a[:, 1:]))
        walls = 2.0 * (np.sum(p[0] ** 2 * inverse_a[0]) + np.sum(p[-1] ** 2 * inverse_a[-1])
                       + np.sum(p[:, 0] ** 2 * inverse_a[:, 0]) + np.sum(p[:, -1] ** 2 * inverse_a[:, -1]))
        energy = float(n * n * (interior_x + interior_y + walls))
    if not np.isfinite(energy) or energy < 0.0:
        raise FloatingPointError("invalid equilibrated-flux energy")
    return energy


def poisson_flux_bounds(defect: np.ndarray, coefficient: np.ndarray) -> FluxBoundResult:
    """Compute the flux and original majorants using one unit-Poisson solve.

    The actual minimum coefficient is obtained directly from the supplied field.
    No reference solution, true error, calibration data, or variable-A inverse is
    required. All arrays must use the same grid/boundary convention as A(a).
    """
    a, d = _matched_fields(coefficient, defect)
    p = solve_poisson(d)
    poisson_energy = float(np.sum(d * p))
    if not np.isfinite(poisson_energy) or poisson_energy < 0.0:
        raise FloatingPointError("invalid Poisson dual energy")
    energy = equilibrated_flux_energy(a, p)
    a_min = float(a.min())
    diagonal = poisson_inverse_diagonal(a.shape[0])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        bound = np.sqrt(diagonal * energy / a_min)
        baseline = np.sqrt(diagonal * poisson_energy) / a_min
    if not np.isfinite(bound).all() or not np.isfinite(baseline).all():
        raise FloatingPointError("nonfinite pointwise flux bound")
    return FluxBoundResult(bound, baseline, p, energy, poisson_energy, a_min)


def equilibrated_flux_error_bound(defect: np.ndarray, coefficient: np.ndarray) -> np.ndarray:
    """Return the tightened pointwise bound, sharing the Poisson-bound API shape."""
    return poisson_flux_bounds(defect, coefficient).bound
