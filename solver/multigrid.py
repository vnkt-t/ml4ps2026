"""AMG-backed rung-4 partial inverse for A e = r.

The preferred path uses PyAMG smoothed aggregation, whose coarsening is built
from the assembled operator and is much less brittle for high-contrast jumps
than the previous geometric block-coarsened V-cycle. If PyAMG is unavailable,
the rung falls back to a deeper Gauss-Seidel partial inverse so the ladder keeps
using the same operator A without the high-contrast geometric-coarsening failure.

COST ACCOUNTING (rewritten 2026-08-21). The previous version built a *two-level*
hierarchy (``max_levels=2``) and charged it a hardcoded 5 matvec-equivalents. That
was wrong in a way that mattered: with only two levels the coarsest operator keeps
~17% of the fine unknowns (176 of 1024 at 32x32) and PyAMG solves it *exactly* with
a dense pseudo-inverse. That coarse solve is O(n_c^3), is NOT counted by PyAMG's
``cycle_complexity()``, and made rung 4 more expensive than the full direct solve it
was supposed to approximate cheaply. A full hierarchy coarsens down to O(1) unknowns,
so the coarse solve is small. ``cycle_complexity()`` remains only a rough
nonzero-count estimate: it omits transfer/residual work and can undercount symmetric
smoothers. It must not be presented as measured work or an exact matvec count.

Two costs are now recorded per call, because they answer different questions:

* ``cycle_complexity`` -- PyAMG's nonzero-based V-cycle work estimate, excluding
  setup. It is useful as an explicitly labeled structural proxy, not equal work.
* measured wall-clock ``setup`` and ``cycle`` times, converted to matvec-equivalents.
  Implementation dependent, but it captures the setup phase, which ``cycle_complexity``
  omits and which CANNOT be amortized here: A depends on the coefficient field a(x), so
  a fresh hierarchy is built for every sample.

Both are exposed via last_vcycle_cost(); reporting only the first would understate the
cost of the rung as actually run.
"""
from __future__ import annotations

import importlib.util
from numbers import Integral
import time
import warnings

import numpy as np
import scipy.sparse as sp

from solver.relaxation import gauss_seidel_sweep


FALLBACK_GS_SWEEPS = 40

# Backend actually used by the most recent vcycle() call. vcycle_backend() reports
# capability (what *would* run); last_vcycle_backend() reports what *did* run, so a
# per-sample AMG->GS fallback is not mislabeled as AMG downstream. (Codex review
# 2026-06-06, Item 3.)
_LAST_VCYCLE_BACKEND = "uninitialized"

# Honest per-call cost record; see module docstring. Populated by vcycle().
_LAST_VCYCLE_COST: dict[str, float | int | str] = {}

# Number of repetitions used to calibrate the reference matvec time. A single matvec is
# ~6 microseconds at 32x32, well below timer resolution, so it must be averaged.
_MATVEC_CALIBRATION_REPS = 50


def pyamg_available() -> bool:
    """Return whether PyAMG can be imported in this environment."""

    return importlib.util.find_spec("pyamg") is not None


def vcycle_backend() -> str:
    """Return the backend rung 4 will use for the current environment (capability)."""

    if pyamg_available():
        return "pyamg_smoothed_aggregation"
    return f"fallback_gs_{FALLBACK_GS_SWEEPS}"


def last_vcycle_backend() -> str:
    """Return the backend the most recent vcycle() call actually used."""

    return _LAST_VCYCLE_BACKEND


def last_vcycle_cost() -> dict[str, float | int | str]:
    """Return the measured cost of the most recent vcycle() call.

    ``cycle_complexity`` is a rough PyAMG work proxy, not measured time.
    ``total_s`` and ``total_matvec_equiv`` include hierarchy setup, the FIRST
    cycle (including lazy coarse factorization), and the acceptance check.
    ``cycle_s`` times a separate warmed cycle for an amortized comparison only.
    Timing calibration and that extra warmed cycle are diagnostic overhead and
    excluded from the one-shot algorithm cost. Failed AMG attempts are charged
    before any GS fallback. ``schema_version=2`` distinguishes older records
    that accidentally omitted the first-cycle cost.
    """

    return dict(_LAST_VCYCLE_COST)


def _time_matvec(A: sp.csr_matrix, x: np.ndarray) -> float:
    """Average seconds for one fine-grid matvec, the unit of the cost axis."""

    t0 = time.perf_counter()
    for _ in range(_MATVEC_CALIBRATION_REPS):
        A @ x
    return (time.perf_counter() - t0) / _MATVEC_CALIBRATION_REPS


def _pyamg_vcycle(
    A: sp.csr_matrix, rhs: np.ndarray, max_levels: int | None
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    import pyamg

    # PyAMG's coarsest-level pinv/matmul can emit benign divide-by-zero / overflow /
    # invalid RuntimeWarnings on tiny, ill-conditioned high-contrast coarse operators.
    # Suppression keeps warnings-as-errors runs (smoke-tests, the OOD sweep) from
    # falsely failing on internal coarse numerics. Correctness is NOT trusted to the
    # absence of warnings: vcycle() independently verifies the cycle reduced the
    # residual and otherwise falls back to GS (Codex review 2026-06-06, Item 3).
    with warnings.catch_warnings(), np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        kwargs = {} if max_levels is None else {"max_levels": int(max_levels)}
        t0 = time.perf_counter()
        ml = pyamg.smoothed_aggregation_solver(A, **kwargs)
        setup_s = time.perf_counter() - t0

        x0 = np.zeros_like(rhs, dtype=np.float64)
        # The first cycle lazily constructs/factorizes the coarse solver. Charge
        # that complete first call in the per-sample total. The extra warmed call
        # below is a benchmark, not work required to produce the returned field.
        t0 = time.perf_counter()
        e = np.asarray(
            ml.solve(rhs, x0=x0.copy(), maxiter=1, cycle="V", tol=0.0, accel=None),
            dtype=np.float64,
        )
        first_cycle_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        ml.solve(rhs, x0=x0.copy(), maxiter=1, cycle="V", tol=0.0, accel=None)
        cycle_s = time.perf_counter() - t0

        matvec_s = _time_matvec(A, rhs)
        total_s = setup_s + first_cycle_s
        cost: dict[str, float | int | str] = {
            "schema_version": 2,
            "cycle_complexity": float(ml.cycle_complexity()),
            "cycle_complexity_kind": "pyamg_nonzero_based_estimate",
            "operator_complexity": float(ml.operator_complexity()),
            "n_levels": int(len(ml.levels)),
            "coarsest_unknowns": int(ml.levels[-1].A.shape[0]),
            "setup_s": float(setup_s),
            "first_cycle_s": float(first_cycle_s),
            "cycle_s": float(cycle_s),
            "total_s": float(total_s),
            "matvec_s": float(matvec_s),
            "setup_matvec_equiv": float(setup_s / matvec_s),
            "first_cycle_matvec_equiv": float(first_cycle_s / matvec_s),
            "cycle_matvec_equiv": float(cycle_s / matvec_s),
            "total_matvec_equiv": float(total_s / matvec_s),
        }
    return e, cost


def _fallback_gs(A: sp.csr_matrix, rhs: np.ndarray) -> np.ndarray:
    e0 = np.zeros_like(rhs, dtype=np.float64)
    return gauss_seidel_sweep(A, rhs, e0, FALLBACK_GS_SWEEPS).reshape(-1)


def _residual_reduced(A: sp.csr_matrix, e: np.ndarray, rhs: np.ndarray) -> bool:
    """Did the partial-inverse e make progress on A e = rhs (vs the e0=0 start)?

    This is an operational acceptance heuristic, not a theorem about a single
    V-cycle: contraction in an energy norm need not imply Euclidean residual
    contraction. The check catches unusable fields for this ladder, not an
    invalid AMG implementation in general.
    """

    if not np.all(np.isfinite(e)):
        return False
    rhs_norm = float(np.linalg.norm(rhs))
    if rhs_norm == 0.0:
        return bool(np.all(e == 0.0))
    return float(np.linalg.norm(A @ e - rhs)) < rhs_norm


def vcycle(A: sp.spmatrix, r: np.ndarray, max_levels: int | None = None) -> np.ndarray:
    """Return the rung-4 approximation to A e = r.

    Uses one PyAMG smoothed-aggregation V-cycle when available, but only if it
    actually reduces the residual; otherwise (or when PyAMG is absent) it returns
    the robust GS-40 partial inverse on the same operator A. The backend actually
    used is recorded in last_vcycle_backend() and the measured cost in
    last_vcycle_cost().

    ``max_levels=None`` (the default) builds a FULL hierarchy, coarsening to O(1)
    unknowns. Passing a small int reproduces the old shallow behaviour and is kept
    only for the resolution/robustness diagnostics -- see the module docstring for
    why a shallow hierarchy misprices this rung.
    """

    global _LAST_VCYCLE_BACKEND, _LAST_VCYCLE_COST
    if max_levels is not None and (
        isinstance(max_levels, bool) or not isinstance(max_levels, Integral) or max_levels < 2
    ):
        raise ValueError(f"max_levels must be None or >= 2; got {max_levels}")
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square; got {A.shape}")

    A_csr = A.tocsr()
    rhs_arr = np.asarray(r, dtype=np.float64)
    rhs = rhs_arr.reshape(-1)
    if rhs.size != A_csr.shape[0]:
        raise ValueError(f"shape mismatch: A {A.shape}, r {rhs.shape}")

    if rhs.size == 0 or not np.all(np.isfinite(rhs)) or not np.all(np.isfinite(A_csr.data)):
        raise ValueError("A and r must be nonempty and finite")
    if np.any(A_csr.diagonal() <= 0.0):
        raise ValueError("A must have a positive diagonal")

    failed_amg_s = 0.0
    failed_amg_complexity = 0.0
    fallback_reason = "pyamg_unavailable"
    if pyamg_available():
        attempt_start = time.perf_counter()
        try:
            e, cost = _pyamg_vcycle(A_csr, rhs, max_levels)
            check_start = time.perf_counter()
            accepted = _residual_reduced(A_csr, e, rhs)
            check_s = time.perf_counter() - check_start
            cost["validation_s"] = check_s
            cost["total_s"] = float(cost["total_s"]) + check_s
            cost["total_matvec_equiv"] = float(cost["total_s"]) / float(cost["matvec_s"])
            if accepted:
                _LAST_VCYCLE_BACKEND = "pyamg_smoothed_aggregation"
                cost["diagnostic_overhead_s"] = max(
                    0.0, time.perf_counter() - attempt_start - float(cost["total_s"])
                )
                _LAST_VCYCLE_COST = cost
                return e.reshape(rhs_arr.shape)
            failed_amg_s = float(cost["total_s"])
            failed_amg_complexity = float(cost["cycle_complexity"])
            fallback_reason = "amg_nonfinite_or_nonreducing"
        except (np.linalg.LinAlgError, RuntimeError, ValueError, FloatingPointError) as exc:
            # A failed setup/solve is still paid for. Preserve the reason in the
            # result so a numerical backend failure cannot silently become AMG.
            failed_amg_s = time.perf_counter() - attempt_start
            fallback_reason = f"{type(exc).__name__}: {exc}"
        _LAST_VCYCLE_BACKEND = f"pyamg_fallback_gs_{FALLBACK_GS_SWEEPS}"
    else:
        _LAST_VCYCLE_BACKEND = f"fallback_gs_{FALLBACK_GS_SWEEPS}"

    t0 = time.perf_counter()
    e = _fallback_gs(A_csr, rhs)
    fallback_s = time.perf_counter() - t0
    if not np.all(np.isfinite(e)):
        raise FloatingPointError("Gauss-Seidel fallback produced a non-finite correction")
    matvec_s = _time_matvec(A_csr, rhs)
    total_s = failed_amg_s + fallback_s
    _LAST_VCYCLE_COST = {
        "schema_version": 2,
        "cycle_complexity": float(FALLBACK_GS_SWEEPS) + failed_amg_complexity,
        "cycle_complexity_kind": "sweep_count_plus_failed_amg_estimate",
        "failed_amg_s": float(failed_amg_s),
        "fallback_s": float(fallback_s),
        "fallback_reason": fallback_reason,
        "total_s": float(total_s),
        "matvec_s": float(matvec_s),
        "total_matvec_equiv": float(total_s / matvec_s),
    }
    return e.reshape(rhs_arr.shape)
