"""Solver-correctness hardening for the Tier-0 gate (beyond the algebraic identity).

test_residual_identity.py is a SELF-CONSISTENCY check: it proves residual.py and
solve.py use the same A, but by construction it passes for ANY invertible A and does
NOT verify that A discretizes -div(a grad u). These tests add that missing
verification:
  - symmetry and positive-definiteness of A (needed for the CG/relaxation ladder);
  - second-order convergence against an ANALYTIC manufactured solution, using a
    coefficient field that is asymmetric under x<->y so a transpose/index bug in the
    assembly would break convergence.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver.assemble_fd import assemble, grid_centers  # noqa: E402
from solver.solve import solve  # noqa: E402

PI = np.pi


def test_symmetry():
    rng = np.random.default_rng(0)
    A = assemble(np.exp(rng.standard_normal((24, 24))))
    asym = abs(A - A.T).max()
    assert asym < 1e-10 * abs(A).max(), f"A not symmetric: max|A-A^T| = {asym:.2e}"


def test_spd():
    rng = np.random.default_rng(1)
    A = assemble(np.exp(rng.standard_normal((16, 16))))
    lam_min = spla.eigsh(A.tocsc(), k=1, which="SA", return_eigenvectors=False)[0]
    assert lam_min > 0, f"A not positive definite: lambda_min = {lam_min:.3e}"


# Manufactured solution: u = sin(pi x) sin(pi y) (exactly 0 on the domain boundary).
# Coefficient a is smooth, strictly positive, and NOT symmetric under x<->y.
def _a_fn(X, Y):
    return 1.0 + 0.5 * np.sin(PI * X) + 0.3 * Y


def _u_fn(X, Y):
    return np.sin(PI * X) * np.sin(PI * Y)


def _f_fn(X, Y):
    # f = -div(a grad u) = -(a_x u_x + a_y u_y + a (u_xx + u_yy))
    a = _a_fn(X, Y)
    a_x = 0.5 * PI * np.cos(PI * X)
    a_y = 0.3 * np.ones_like(Y)
    u_x = PI * np.cos(PI * X) * np.sin(PI * Y)
    u_y = PI * np.sin(PI * X) * np.cos(PI * Y)
    lap = -2.0 * PI * PI * np.sin(PI * X) * np.sin(PI * Y)
    return -(a_x * u_x + a_y * u_y + a * lap)


def _l2_error(N: int) -> float:
    X, Y = grid_centers(N)
    A = assemble(_a_fn(X, Y))
    u_h = solve(A, _f_fn(X, Y)).reshape(N, N)
    h = 1.0 / N
    return float(np.sqrt(np.sum((u_h - _u_fn(X, Y)) ** 2) * h * h))


def convergence_table(Ns=(16, 32, 64, 128)):
    errs = [_l2_error(N) for N in Ns]
    rates = [np.log(errs[i] / errs[i + 1]) / np.log(2) for i in range(len(Ns) - 1)]
    return list(Ns), errs, rates


def test_manufactured_convergence():
    Ns, errs, rates = convergence_table()
    assert errs[-1] < 1e-3, f"L2 error too large at N={Ns[-1]}: {errs[-1]:.2e}"
    # finest-grid rate should approach 2 for smooth-coefficient TPFA
    assert rates[-1] > 1.7, f"convergence slower than ~2: finest rate = {rates[-1]:.2f}"


if __name__ == "__main__":
    Ns, errs, rates = convergence_table()
    print("manufactured convergence (u = sin(pi x) sin(pi y), asymmetric a):")
    print(f"{'N':>5} {'L2 error':>14} {'rate':>8}")
    for k, N in enumerate(Ns):
        rate = "" if k == 0 else f"{rates[k-1]:.2f}"
        print(f"{N:>5} {errs[k]:>14.4e} {rate:>8}")
    ok = errs[-1] < 1e-3 and rates[-1] > 1.7
    print("PASS" if ok else "FAIL", f"(finest rate {rates[-1]:.2f}, target ~2)")
    sys.exit(0 if ok else 1)
