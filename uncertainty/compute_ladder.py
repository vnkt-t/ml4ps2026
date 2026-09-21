"""Assemble all compute-ladder rungs for one test sample."""
from __future__ import annotations

import time
from collections import OrderedDict

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import gaussian_filter

from solver.multigrid import vcycle, last_vcycle_backend, last_vcycle_cost
from solver.relaxation import cg_correction, gauss_seidel_sweep, jacobi_sweep
from uncertainty.ensemble_scores import ensemble_std
from uncertainty.residual_scores import infer_field_shape, residual_vector


def _as_field(x: np.ndarray | float | None, shape: tuple[int, int]) -> np.ndarray | None:
    if x is None:
        return None
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0 or arr.size == 1:
        return np.full(shape, float(arr.reshape(-1)[0]), dtype=np.float64)
    return arr.reshape(shape)


def compute_ladder(
    A: sp.spmatrix,
    u_hat: np.ndarray,
    f: np.ndarray | float,
    u_test: np.ndarray | None,
    u_pred_ensemble: np.ndarray | None,
    sigma_ens: np.ndarray | None,
) -> dict[str, dict[str, np.ndarray | float | int]]:
    """Compute score fields and simple cost accounting for rungs 0 through 5."""
    shape = infer_field_shape(A, u_hat, f, u_test if u_test is not None else u_hat)

    t0 = time.perf_counter()
    if sigma_ens is not None:
        sigma = _as_field(sigma_ens, shape)
    elif u_pred_ensemble is not None:
        sigma = ensemble_std(np.asarray(u_pred_ensemble), axis=0).reshape(shape)
    else:
        sigma = np.zeros(shape, dtype=np.float64)
    sigma_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    r_vec = residual_vector(A, u_hat, f)
    residual_time = time.perf_counter() - t0
    r_field = r_vec.reshape(shape)
    abs_r = np.abs(r_field)

    out: OrderedDict[str, dict[str, np.ndarray | float | int]] = OrderedDict()
    out["0_sigma_ens"] = {"score": sigma, "time_s": sigma_time, "matvecs": 0}
    out["1_abs_r"] = {"score": abs_r, "time_s": residual_time, "matvecs": 1}

    t0 = time.perf_counter()
    smoothed = gaussian_filter(abs_r, sigma=1.0, mode="nearest")
    out["2_smooth_abs_r"] = {
        "score": smoothed,
        "time_s": residual_time + (time.perf_counter() - t0),
        "matvecs": 1,
    }

    e0 = np.zeros(shape, dtype=np.float64)
    for label, sweeps in [
        ("3a_jacobi_1", 1),
        ("3b_jacobi_2", 2),
        ("3c_jacobi_5", 5),
        ("3d_jacobi_10", 10),
        ("3e_jacobi_20", 20),
    ]:
        t0 = time.perf_counter()
        e_k = jacobi_sweep(A, r_vec, e0, sweeps)
        out[label] = {
            "score": np.abs(e_k.reshape(shape)),
            "time_s": residual_time + (time.perf_counter() - t0),
            "matvecs": 1 + sweeps,
        }

    for label, sweeps in [("3f_gs_1", 1), ("3g_gs_5", 5)]:
        t0 = time.perf_counter()
        e_k = gauss_seidel_sweep(A, r_vec, e0, sweeps)
        out[label] = {
            "score": np.abs(e_k.reshape(shape)),
            "time_s": residual_time + (time.perf_counter() - t0),
            "matvecs": 1 + sweeps,
        }

    # CG is A-norm optimal in its Krylov space (different from Jacobi's space).
    # From a zero start, its sparse-residual SUPPORT expands at most k-1 hops.
    # Global dot products nevertheless make each iterate depend on distant
    # residual VALUES. Do not equate that support bound with a local estimator.
    for label, iters in [("3h_cg_5", 5), ("3i_cg_10", 10), ("3j_cg_20", 20)]:
        t0 = time.perf_counter()
        e_cg, cg_info = cg_correction(A, r_vec, maxiter=iters)
        out[label] = {
            "score": np.abs(np.asarray(e_cg, dtype=np.float64).reshape(shape)),
            "time_s": residual_time + (time.perf_counter() - t0),
            # one matvec per CG iteration, plus the residual apply
            "matvecs": 1 + int(cg_info["iterations_performed"]),
            **cg_info,
        }

    t0 = time.perf_counter()
    e_v = vcycle(A, r_vec)
    vcycle_cost = last_vcycle_cost()
    out["4_vcycle"] = {
        "score": np.abs(e_v.reshape(shape)),
        # The vcycle call also benchmarks a warmed cycle and calibrates a
        # matvec timer. Exclude that diagnostic overhead from one-shot time.
        "time_s": residual_time + float(vcycle_cost["total_s"]),
        "instrumented_call_time_s": residual_time + (time.perf_counter() - t0),
        # Two distinct cost quantities, because they answer
        # different questions (see solver/multigrid.py docstring):
        #   matvecs             -> PyAMG's rough nnz-based cycle work estimate
        #   matvecs_with_setup  -> cost as ACTUALLY run, including the per-sample AMG
        #                          setup, which cannot be amortized because A depends
        #                          on a(x). At 32x32/64x64 this exceeds a direct solve.
        "matvecs": 1 + float(vcycle_cost.get("cycle_complexity", float("nan"))),
        "matvecs_with_setup": 1 + float(vcycle_cost.get("total_matvec_equiv", float("nan"))),
        "amg_cost": vcycle_cost,
        # Report the backend that ACTUALLY ran this sample (AMG vs per-sample GS
        # fallback), not just the capability. (Codex review 2026-06-06, Item 3.)
        "backend": last_vcycle_backend(),
    }

    t0 = time.perf_counter()
    A_csc = A.tocsc()
    e_exact = np.asarray(spla.spsolve(A_csc, r_vec), dtype=np.float64)
    oracle_s = time.perf_counter() - t0
    # Measure the oracle in the SAME matvec unit as rung 4, so the two are comparable.
    # Without this there is no way to check the claim that a partial inverse is cheaper
    # than the exact solve it approximates -- which at these grid sizes it is not.
    t0 = time.perf_counter()
    for _ in range(50):
        A @ r_vec
    matvec_s = (time.perf_counter() - t0) / 50
    out["5_oracle"] = {
        "score": np.abs(e_exact.reshape(shape)),
        "time_s": residual_time + oracle_s,
        "matvecs_with_setup": 1.0 + float(oracle_s / matvec_s) if matvec_s > 0 else float("nan"),
        # NOTE: the oracle is a FULL DIRECT SOLVE, not an iterative method. "matvecs": 1
        # is a placeholder (the one residual apply) and is NOT a meaningful iterative cost
        # proxy — do NOT plot localization-vs-matvecs with the oracle at cost=1. Treat it as
        # the off-scale "full solve" endpoint. (Codex review 2026-06-06.)
        "matvecs": 1,
        "cost_class": "full_solve_off_matvec_scale",
    }

    # Stamp an explicit cost class on every rung so a localization-vs-cost figure can label the
    # compute axis honestly (matvecs alone is misleading: oracle is a full solve, sigma_ens is
    # ensemble forward passes, neither is "1 cheap operator apply"). (Cost taxonomy, 2026-06-06.)
    for name, payload in out.items():
        if "cost_class" in payload:
            continue  # oracle already stamped above
        if name == "0_sigma_ens":
            payload["cost_class"] = "ensemble_forward_passes"
        elif name == "1_abs_r":
            payload["cost_class"] = "one_operator_apply"
        elif name == "2_smooth_abs_r":
            payload["cost_class"] = "operator_apply_plus_smoothing"
        elif "_cg_" in name:  # single-level Krylov (checked before the "3" branch)
            payload["cost_class"] = "krylov_iterations"
        elif name.startswith("3"):  # Jacobi / Gauss-Seidel relaxation rungs
            payload["cost_class"] = "relaxation_sweeps"
        elif name == "4_vcycle":
            payload["cost_class"] = "multilevel_partial_inverse"
        else:
            payload["cost_class"] = "unspecified"
    return out
