"""Is the A-norm-OPTIMAL bounded-radius correction also the better error localizer?

CG from a zero start produces x_k = p_{k-1}(A) r and is A-norm optimal over the Krylov space
K_k(A, r). Jacobi is NOT in that space: from zero it gives sum_{j<k} (I - D^-1 A)^j D^-1 r, a
polynomial in the iteration matrix D^-1 A applied to D^-1 r, and the spatially varying diagonal
D does not commute with A for a heterogeneous coefficient. The two share a finite propagation
radius (D^-1 is pointwise; each step applies A once), not an approximation space.

The comparison below therefore does NOT rest on a shared space. It rests on what CG is built
to do: be accurate in the A-norm. It succeeds at that and still loses at localization.

If CG-20 has the smaller A-norm error but the WORSE Spearman against |e|, that is a clean
instance of the norm mismatch the paper documents (H2): being closer in the energy norm does
not make a field a better ranker of pointwise error magnitude. This script measures both.

Usage: .venv/bin/python analysis/check_krylov_anorm.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import scipy.sparse.linalg as spla
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from solver.assemble_fd import assemble  # noqa: E402
from solver.relaxation import jacobi_sweep  # noqa: E402

ITERS = 20


def a_norm(A, v: np.ndarray) -> float:
    # SciPy's sparse matmul can emit benign divide/overflow/invalid RuntimeWarnings on these
    # high-contrast operators; the quadratic form itself is finite and is asserted below.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        q = float(v @ (A @ v))
    if not np.isfinite(q):
        raise FloatingPointError("non-finite A-norm quadratic form")
    return float(np.sqrt(max(q, 0.0)))


def main() -> None:
    d = np.load(os.path.join(REPO, "results", "fno_ensemble_kappa100.npz"), allow_pickle=True)
    a_test, u_test, u_pred, f_test = d["a_test"], d["u_test"], d["u_pred"], d["f_test"]
    n = a_test.shape[0]

    rows = {k: [] for k in ("jac_rho", "cg_rho", "jac_anorm", "cg_anorm")}
    for i in range(n):
        A = assemble(a_test[i]).tocsr()
        u_hat = u_pred[:, i].astype(np.float64).mean(0)
        e = (u_hat - u_test[i]).reshape(-1)
        r = A @ u_hat.reshape(-1) - f_test[i].reshape(-1)
        abs_e = np.abs(e)

        x_j = jacobi_sweep(A, r, np.zeros_like(r), ITERS).reshape(-1)
        x_c, _ = spla.cg(A, r, x0=np.zeros_like(r), rtol=0.0, atol=0.0, maxiter=ITERS)
        x_c = np.asarray(x_c, dtype=np.float64)

        rows["jac_rho"].append(spearmanr(np.abs(x_j), abs_e).statistic)
        rows["cg_rho"].append(spearmanr(np.abs(x_c), abs_e).statistic)
        # relative A-norm error of the correction against the true error field
        denom = a_norm(A, e)
        rows["jac_anorm"].append(a_norm(A, x_j - e) / denom)
        rows["cg_anorm"].append(a_norm(A, x_c - e) / denom)

    out = {
        "kappa": 100.0,
        "grid_N": int(a_test.shape[1]),
        "n_test": int(n),
        "iterations": ITERS,
        "note": "different approximation spaces; shared finite propagation radius (<= k-1 hops)",
        **{k: float(np.mean(v)) for k, v in rows.items()},
    }
    cg_better_anorm = out["cg_anorm"] < out["jac_anorm"]
    cg_worse_rho = out["cg_rho"] < out["jac_rho"]
    out["cg_better_in_A_norm"] = bool(cg_better_anorm)
    out["cg_worse_at_localization"] = bool(cg_worse_rho)
    out["norm_mismatch_demonstrated"] = bool(cg_better_anorm and cg_worse_rho)

    path = os.path.join(REPO, "results", "check_krylov_anorm.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"kappa=100, {n} test fields, {ITERS} iterations, same propagation radius")
    print(f"  Jacobi-20  rel A-norm err {out['jac_anorm']:.4f}   Spearman {out['jac_rho']:.4f}")
    print(f"  CG-20      rel A-norm err {out['cg_anorm']:.4f}   Spearman {out['cg_rho']:.4f}")
    print(f"  CG better in A-norm: {cg_better_anorm}    CG worse at localization: {cg_worse_rho}")
    print(f"  -> norm mismatch demonstrated: {out['norm_mismatch_demonstrated']}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
