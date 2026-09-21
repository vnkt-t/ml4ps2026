"""Stop Poisson-PCG using computable top-error membership enclosures.

The declared default rule checks every five iterations and stops when strict
interval separation identifies at least 90% of the top ceil(10% * n) cells.
The solver takes only the operator, residual, and a coefficient lower bound.
It never receives a true solution, true error, or oracle correction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg

from solver.poisson import solve_poisson
from uncertainty.flux_bounds import equilibrated_flux_error_bound
from uncertainty.rank_bounds import magnitude_envelope, poisson_error_bound, separate_top_set


@dataclass(frozen=True)
class AdaptiveRankResult:
    correction: np.ndarray
    bound: np.ndarray
    high: np.ndarray
    low: np.ndarray
    iterations: int
    status: str
    target_met: bool
    certified_high_recall: float
    n_top: int
    target_high_count: int
    bound_kind: str
    cg_operator_applications: int
    extra_operator_applications: int
    preconditioner_applications: int
    bound_poisson_solves: int
    fft_transforms: int
    history: tuple[dict[str, int | float], ...]


class _RankTargetReached(Exception):
    """Private control flow to leave SciPy's callback without a wasted restart."""


def adaptive_poisson_cg(
    A: sp.spmatrix,
    r: np.ndarray,
    a_min: float,
    *,
    q: float = 0.1,
    target_recall: float = 0.9,
    maxiter: int = 80,
    check_every: int = 5,
    bound_kind: str = "poisson",
    coefficient: np.ndarray | None = None,
) -> AdaptiveRankResult:
    """Return correction and rank enclosure after an untuned stopping rule.

    ``r`` may be a square field or a vector of N² values. Return arrays have
    the same shape as ``r``. ``A`` must be the SPD TPFA operator on the matching
    grid, and ``a_min`` a valid positive lower coefficient bound.

    ``bound_kind='flux'`` uses the classical equilibrated-flux tightening and
    requires the matching coefficient field. It uses the actual field minimum;
    the supplied ``a_min`` is still checked as a valid lower bound on that field.
    Both variants require one Poisson solve per enclosure check.

    A target is met only when certified-high count >= ceil(target_recall*k),
    k=ceil(q*N²). A valid enclosure makes this count a lower bound on true
    top-set recall; no true-error input or calibrated threshold is needed.
    Exact linear convergence with unresolved ties returns
    ``converged_unresolved``. Exhausting maxiter returns ``iteration_limit``.

    The enclosure theorem is exact-arithmetic/discrete; its float64 evaluation
    does not provide outward rounding. Counts include the extra residual apply
    and Poisson solve at every enclosure check. The reusable Green diagonal's
    O(N³) setup is not an FFT and is not included in ``fft_transforms``. That
    counter denotes forward/inverse 2D DST calls (two calls per Poisson solve).
    """
    for name, value in (("maxiter", maxiter), ("check_every", check_every)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(q) or not 0 < q <= 1 or not np.isfinite(target_recall) or not 0 < target_recall <= 1:
        raise ValueError("q and target_recall must be in (0,1]")
    if not np.isfinite(a_min) or a_min <= 0:
        raise ValueError("a_min must be finite and positive")
    if bound_kind not in ("poisson", "flux"):
        raise ValueError("bound_kind must be 'poisson' or 'flux'")
    rhs_arr = np.asarray(r, dtype=np.float64)
    rhs = rhs_arr.reshape(-1)
    n = int(np.sqrt(rhs.size))
    matrix = A.tocsr().astype(np.float64, copy=False)
    if rhs.size == 0 or n*n != rhs.size or matrix.shape != (rhs.size, rhs.size):
        raise ValueError("A and r must describe a nonempty square N-by-N grid")
    if rhs_arr.ndim == 2 and rhs_arr.shape != (n, n):
        raise ValueError("a two-dimensional r must have square field shape")
    if not np.isfinite(rhs).all() or not np.isfinite(matrix.data).all() or np.any(matrix.diagonal() <= 0):
        raise ValueError("A and r must be finite and A must have a positive diagonal")
    coefficients = None
    if coefficient is not None:
        coefficients = np.asarray(coefficient, dtype=np.float64)
        if coefficients.shape != (n, n) or not np.isfinite(coefficients).all() or np.any(coefficients <= 0):
            raise ValueError("coefficient must be a matching finite positive square field")
        if a_min > float(coefficients.min()):
            raise ValueError("a_min exceeds the supplied coefficient minimum")
    if bound_kind == "flux" and coefficients is None:
        raise ValueError("the flux bound requires the matching coefficient field")

    n_top = int(np.ceil(q * rhs.size))
    target_high_count = int(np.ceil(target_recall * n_top))
    iterations = 0
    cg_applies = 0
    extra_applies = 0
    preconditioner_applies = 0
    checks = 0
    last_checked = -1
    latest = np.zeros_like(rhs)
    bound = np.zeros((n, n))
    high = np.zeros((n, n), dtype=bool)
    low = np.zeros((n, n), dtype=bool)
    history = []

    def apply_operator(vector):
        nonlocal cg_applies
        cg_applies += 1
        return matrix @ vector

    def apply_preconditioner(vector):
        nonlocal preconditioner_applies
        preconditioner_applies += 1
        return solve_poisson(np.asarray(vector).reshape(n, n)).reshape(-1)

    def assess(candidate):
        nonlocal checks, extra_applies, bound, high, low, latest, last_checked
        latest = np.asarray(candidate).copy()
        if not np.isfinite(latest).all():
            raise FloatingPointError("Poisson-PCG produced a non-finite correction")
        defect = (rhs - matrix @ latest).reshape(n, n)
        extra_applies += 1
        bound = (equilibrated_flux_error_bound(defect, coefficients)
                 if bound_kind == "flux" else poisson_error_bound(defect, a_min))
        checks += 1
        lower, upper = magnitude_envelope(latest.reshape(n, n), bound)
        high, low = separate_top_set(lower, upper, q)
        last_checked = iterations
        history.append({"iterations": iterations, "certified_high_count": int(high.sum()),
                        "certified_high_recall": float(high.sum() / n_top),
                        "resolved_fraction": float(np.mean(high | low))})
        return int(high.sum()) >= target_high_count

    def callback(candidate):
        nonlocal iterations
        iterations += 1
        if iterations % check_every == 0 and assess(candidate):
            raise _RankTargetReached()

    if not np.any(rhs) or n_top == rhs.size:
        target_met = assess(latest)
        status = "rank_target" if target_met else "converged_unresolved"
    else:
        operator = LinearOperator(matrix.shape, matvec=apply_operator, dtype=np.float64)
        preconditioner = LinearOperator(matrix.shape, matvec=apply_preconditioner, dtype=np.float64)
        try:
            candidate, info = cg(
                operator, rhs, x0=np.zeros_like(rhs), M=preconditioner, maxiter=int(maxiter),
                atol=0.0, rtol=8.0 * np.finfo(np.float64).eps, callback=callback,
            )
        except _RankTargetReached:
            target_met = True
            status = "rank_target"
        else:
            if info < 0 or not np.isfinite(candidate).all():
                raise FloatingPointError(f"Poisson-PCG failed with info={info}")
            target_met = assess(candidate) if last_checked != iterations else int(high.sum()) >= target_high_count
            status = "rank_target" if target_met else ("converged_unresolved" if info == 0 else "iteration_limit")

    return AdaptiveRankResult(
        correction=latest.reshape(rhs_arr.shape), bound=bound.reshape(rhs_arr.shape),
        high=high.reshape(rhs_arr.shape), low=low.reshape(rhs_arr.shape),
        iterations=iterations, status=status, target_met=target_met,
        certified_high_recall=float(high.sum() / n_top), n_top=n_top,
        target_high_count=target_high_count, bound_kind=bound_kind, cg_operator_applications=cg_applies,
        extra_operator_applications=extra_applies,
        preconditioner_applications=preconditioner_applies, bound_poisson_solves=checks,
        fft_transforms=2 * (preconditioner_applies + checks), history=tuple(history),
    )
