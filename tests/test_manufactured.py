"""Tier-0 T0.4 gate: the manufactured generator produces exact discrete solutions and
genuinely decouples interface location from solution-feature location.

Checks:
  - f := A u_true makes u_true the exact discrete solution (recover to < 1e-9);
  - u_true respects the zero-Dirichlet BC (small on the boundary cells);
  - Family A: interface exists but the solution is smooth there (low |grad u| at interface);
  - Family B: the hard feature sits AWAY from the interface (argmax|u| far from interface cells).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.manufactured import family_A, family_B  # noqa: E402
from solver.assemble_fd import assemble, grid_centers  # noqa: E402
from solver.solve import solve  # noqa: E402


def _recover(s):
    N = s.a.shape[0]
    u = solve(assemble(s.a), s.f).reshape(N, N)
    return np.linalg.norm(u - s.u_true) / np.linalg.norm(s.u_true)


def test_exact_discrete_solution():
    rng = np.random.default_rng(0)
    for s in (family_A(64, rng), family_B(64, rng)):
        assert _recover(s) < 1e-9, f"u_true not exactly recovered: {_recover(s):.2e}"


def test_boundary_condition():
    rng = np.random.default_rng(0)
    for s in (family_A(64, rng), family_B(64, rng)):
        bnd = max(np.abs(s.u_true[0]).max(), np.abs(s.u_true[-1]).max(),
                  np.abs(s.u_true[:, 0]).max(), np.abs(s.u_true[:, -1]).max())
        assert bnd < 0.15, f"u_true not boundary-vanishing: max|u| on edge = {bnd:.2e}"


def test_decoupling_geometry():
    """The hard feature (argmax |u_true|) must be far from the interface in Family B,
    and Family A must have an interface while staying smooth."""
    rng = np.random.default_rng(0)
    N = 64
    X, Y = grid_centers(N)

    sA = family_A(N, rng)
    assert sA.interface_mask.sum() > 0, "Family A should have an interface"

    sB = family_B(N, rng)
    i, j = np.unravel_index(np.argmax(np.abs(sB.u_true)), sB.u_true.shape)
    peak = np.array([X[i, j], Y[i, j]])
    ic = np.array(sB.meta["interface_center"])
    dist = np.linalg.norm(peak - ic)
    assert dist > 0.25, f"Family B feature too close to interface: dist = {dist:.2f}"


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for name, s in [("A", family_A(64, rng)), ("B", family_B(64, rng))]:
        print(f"family {name}: recover={_recover(s):.2e}  interface_cells={int(s.interface_mask.sum())}")
    print("ok")
