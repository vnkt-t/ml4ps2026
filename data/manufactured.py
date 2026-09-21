"""Manufactured solutions (Family M) for the interface/error decoupling experiments (Exp 4).

Idea: pick a coefficient field a(x) and a target solution u_true(x), then DEFINE the source
so u_true is the exact discrete solution:  f = A(a) @ u_true. Solving A u = f then recovers
u_true to solver precision. This lets us place the high-contrast INTERFACE and the hard-to-fit
SOLUTION FEATURE independently — exactly what Experiment 4 needs to show that a residual can
light up the interface while the real error lives elsewhere.

Composable pieces
-----------------
coefficient (where the interface is):
  coeff_smooth(N, rng)                      smooth positive GRF-like field
  coeff_inclusion(N, center, radius, ...)   high-contrast square inclusion at `center`
solution (where the hard feature is):
  sol_smooth(N)                             low-frequency, boundary-vanishing
  sol_bump(N, center, width, freq, amp)     localized high-frequency feature at `center`

Decoupling families (plan Exp 4):
  family_A: inclusion + smooth solution            -> interface present, low error there
  family_B: inclusion + bump placed AWAY from it   -> error concentrated away from interface
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver.assemble_fd import assemble, grid_centers  # noqa: E402


@dataclass
class Sample:
    """One manufactured instance. u_true is the exact discrete solution of A u = f."""
    a: np.ndarray          # (N, N) coefficient field
    u_true: np.ndarray     # (N, N) target solution
    f: np.ndarray          # (N, N) source, = (A @ u_true) reshaped
    interface_mask: np.ndarray  # (N, N) bool, True on high-contrast interface cells (or all-False)
    meta: dict


def _envelope(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Boundary-vanishing envelope (=0 on the [0,1]^2 walls)."""
    return np.sin(np.pi * X) * np.sin(np.pi * Y)


# ---------------------------------------------------------------- coefficients
def coeff_smooth(N: int, rng: np.random.Generator, lengthscale: float = 16.0,
                 sigma: float = 0.5) -> np.ndarray:
    g = gaussian_filter(rng.standard_normal((N, N)), sigma=N / lengthscale)
    g = sigma * (g - g.mean()) / (g.std() + 1e-12)
    return np.exp(g)


def coeff_inclusion(N: int, center=(0.3, 0.3), half=0.12, a_low: float = 1.0,
                    contrast: float = 100.0):
    """High-contrast square inclusion (a_high = a_low*contrast) centered at `center`.

    Returns (a, interface_mask) where the mask marks cells on the inclusion boundary.
    """
    X, Y = grid_centers(N)
    cx, cy = center
    inside = (np.abs(X - cx) <= half) & (np.abs(Y - cy) <= half)
    a = np.full((N, N), a_low, dtype=np.float64)
    a[inside] = a_low * contrast
    # interface = boundary of the inclusion (cells adjacent to a contrast jump)
    mask = np.zeros((N, N), dtype=bool)
    mask[:-1, :] |= inside[:-1, :] != inside[1:, :]
    mask[1:, :] |= inside[:-1, :] != inside[1:, :]
    mask[:, :-1] |= inside[:, :-1] != inside[:, 1:]
    mask[:, 1:] |= inside[:, :-1] != inside[:, 1:]
    return a, mask


# ------------------------------------------------------------------- solutions
def sol_smooth(N: int, amp: float = 1.0) -> np.ndarray:
    X, Y = grid_centers(N)
    return amp * _envelope(X, Y)


def sol_bump(N: int, center=(0.7, 0.7), width: float = 0.07, freq: float = 8.0,
             amp: float = 1.0) -> np.ndarray:
    """Localized high-frequency feature at `center`, times the boundary envelope."""
    X, Y = grid_centers(N)
    cx, cy = center
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    bump = np.exp(-r2 / (2.0 * width ** 2)) * np.cos(2.0 * np.pi * freq * (X - cx))
    return amp * bump * _envelope(X, Y)


# ----------------------------------------------------------- assemble a sample
def make_sample(a: np.ndarray, u_true: np.ndarray, interface_mask=None,
                meta: dict | None = None) -> Sample:
    """Define f := A(a) @ u_true so u_true is the exact discrete solution."""
    N = a.shape[0]
    A = assemble(a)
    f = (A @ u_true.reshape(-1)).reshape(N, N)
    if interface_mask is None:
        interface_mask = np.zeros((N, N), dtype=bool)
    return Sample(a=a, u_true=u_true, f=f, interface_mask=interface_mask, meta=meta or {})


# --------------------------------------------------------- decoupling families
def family_A(N: int, rng: np.random.Generator, contrast: float = 100.0,
             interface_center=(0.3, 0.3)) -> Sample:
    """Interface present, SMOOTH solution -> low true error at the interface."""
    a, mask = coeff_inclusion(N, center=interface_center, contrast=contrast)
    u = sol_smooth(N)
    return make_sample(a, u, mask, meta=dict(family="A", contrast=contrast,
                                             interface_center=interface_center))


def family_B(N: int, rng: np.random.Generator, contrast: float = 100.0,
             interface_center=(0.3, 0.3), bump_center=(0.7, 0.7)) -> Sample:
    """Interface at one location, HARD solution feature elsewhere -> error away from interface."""
    a, mask = coeff_inclusion(N, center=interface_center, contrast=contrast)
    u = sol_smooth(N, amp=0.3) + sol_bump(N, center=bump_center)
    return make_sample(a, u, mask, meta=dict(family="B", contrast=contrast,
                                             interface_center=interface_center,
                                             bump_center=bump_center))


if __name__ == "__main__":
    from solver.solve import solve

    N = 64
    rng = np.random.default_rng(0)
    for name, s in [("A", family_A(N, rng)), ("B", family_B(N, rng))]:
        u_rec = solve(assemble(s.a), s.f).reshape(N, N)
        rel = np.linalg.norm(u_rec - s.u_true) / np.linalg.norm(s.u_true)
        bnd = max(np.abs(s.u_true[0]).max(), np.abs(s.u_true[-1]).max(),
                  np.abs(s.u_true[:, 0]).max(), np.abs(s.u_true[:, -1]).max())
        print(f"family {name}: recover ||u-u_true||/||u_true|| = {rel:.2e} | "
              f"max |u_true| on boundary row/col = {bnd:.2e} | "
              f"interface cells = {int(s.interface_mask.sum())}")
