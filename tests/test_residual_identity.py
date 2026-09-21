"""Experiment 1 gate: verify the exact discrete error identity  e = A^{-1} r.

This must pass before any scientific claim. It confirms residual.py uses the SAME
discrete operator as solve.py — the one consistency the plan calls non-negotiable.

Procedure (per sample): assemble A(a); solve A u = f; form a synthetic prediction
u_hat = u + perturbation; compute r = A u_hat - f; solve A z = r; check z == (u_hat - u).
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver.assemble_fd import assemble  # noqa: E402
from solver.residual import residual  # noqa: E402
from solver.solve import solve  # noqa: E402

TOL = 1e-8


def coeff_field(N: int, rng: np.random.Generator, contrast: float = 10.0) -> np.ndarray:
    """Smooth positive field with a high-contrast inclusion (stresses conditioning)."""
    g = gaussian_filter(rng.standard_normal((N, N)), sigma=N / 16.0)
    a = np.exp(g - g.mean())
    a[N // 4 : N // 2, N // 4 : N // 2] *= contrast
    return a


def run(N: int = 32, seed: int = 0, contrast: float = 10.0):
    rng = np.random.default_rng(seed)
    a = coeff_field(N, rng, contrast)
    A = assemble(a)

    f = gaussian_filter(rng.standard_normal((N, N)), sigma=N / 24.0).reshape(-1)
    u = solve(A, f)

    pert = 0.1 * (np.abs(u).mean() + 1e-12) * rng.standard_normal(N * N)
    u_hat = u + pert
    e = u_hat - u

    r = residual(A, u_hat, f)
    z = solve(A, r)

    id_err = np.linalg.norm(z - e) / np.linalg.norm(e)
    res_err = np.linalg.norm(A.dot(z) - r) / np.linalg.norm(r)
    return id_err, res_err


def test_residual_identity():
    for seed in range(5):
        id_err, res_err = run(N=32, seed=seed)
        assert id_err < TOL, f"identity ||z-e||/||e|| = {id_err:.2e} >= {TOL} (seed {seed})"
        assert res_err < TOL, f"residual ||Az-r||/||r|| = {res_err:.2e} >= {TOL} (seed {seed})"


if __name__ == "__main__":
    print(f"{'seed':>4} {'||z-e||/||e||':>16} {'||Az-r||/||r||':>16}  result")
    all_ok = True
    for seed in range(5):
        ie, re_ = run(N=32, seed=seed)
        ok = ie < TOL and re_ < TOL
        all_ok &= ok
        print(f"{seed:>4} {ie:>16.3e} {re_:>16.3e}  {'PASS' if ok else 'FAIL'}")
    print("ALL PASS" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)
