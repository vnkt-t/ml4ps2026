"""Fairness check: is the paper's UNDAMPED Jacobi rung handicapping the local baseline?

Damped Jacobi (omega = 2/3) is the standard multigrid smoother; undamped Jacobi is a poor
smoother for the oscillatory modes. A reviewer can reasonably ask whether rung 3 was set up
to fail. This runs both at 20 sweeps over the full kappa=100 test set and reports the pair.

Usage: .venv/bin/python analysis/check_damped_jacobi.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from solver.assemble_fd import assemble  # noqa: E402

SWEEPS = 20
OMEGA = 2.0 / 3.0


def main() -> None:
    d = np.load(os.path.join(REPO, "results", "fno_ensemble_kappa100.npz"), allow_pickle=True)
    a_test, u_test, u_pred, f_test = d["a_test"], d["u_test"], d["u_pred"], d["f_test"]
    n = a_test.shape[0]

    undamped, damped = [], []
    for i in range(n):
        A = assemble(a_test[i]).tocsr()
        u_hat = u_pred[:, i].astype(np.float64).mean(0)
        abs_e = np.abs((u_hat - u_test[i]).reshape(-1))
        r = A @ u_hat.reshape(-1) - f_test[i].reshape(-1)
        d_inv = 1.0 / A.diagonal()

        for omega, sink in ((1.0, undamped), (OMEGA, damped)):
            x = np.zeros_like(r)
            for _ in range(SWEEPS):
                x = x + omega * d_inv * (r - A @ x)
            sink.append(spearmanr(np.abs(x), abs_e).statistic)

    out = {
        "kappa": 100.0,
        "grid_N": int(a_test.shape[1]),
        "n_test": int(n),
        "sweeps": SWEEPS,
        "undamped_jacobi_spearman_mean": float(np.mean(undamped)),
        "damped_jacobi_omega": OMEGA,
        "damped_jacobi_spearman_mean": float(np.mean(damped)),
    }
    path = os.path.join(REPO, "results", "check_damped_jacobi.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"kappa=100, {n} test fields, {SWEEPS} sweeps")
    print(f"  undamped Jacobi      rho = {out['undamped_jacobi_spearman_mean']:.4f}")
    print(f"  damped Jacobi (2/3)  rho = {out['damped_jacobi_spearman_mean']:.4f}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
