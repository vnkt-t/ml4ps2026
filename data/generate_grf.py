"""Smooth log-Gaussian-random-field coefficients (Family S) + solved Darcy datasets.

a(x) = exp(sigma * g(x)), g a smooth GRF (filtered white noise), strictly positive. The
training data solves -div(a grad u) = f with a FIXED, smooth source f INDEPENDENT of a
(standard Darcy setup) — so f carries no interface structure and the FNO's error is a
realistic learned-error field (avoids the manufactured f=A·u_true source-spike confound).
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver.assemble_fd import assemble  # noqa: E402
from solver.solve import solve  # noqa: E402


def grf_coeff(N: int, rng: np.random.Generator, lengthscale: float = 0.12,
              sigma: float = 1.0) -> np.ndarray:
    """Smooth positive log-GRF coefficient on an N x N grid."""
    sig_px = max(lengthscale * N, 0.5)
    g = gaussian_filter(rng.standard_normal((N, N)), sigma=sig_px, mode="wrap")
    g = (g - g.mean()) / (g.std() + 1e-12)
    return np.exp(sigma * g)


def make_dataset(N: int, n_samples: int, rng: np.random.Generator, source: float = 1.0,
                 lengthscale: float = 0.12, sigma: float = 1.0):
    """Return (a, u) arrays of shape (n_samples, N, N); u solves -div(a grad u)=source."""
    f = np.full((N, N), float(source), dtype=np.float64)
    A = np.empty((n_samples, N, N), dtype=np.float64)
    U = np.empty((n_samples, N, N), dtype=np.float64)
    for k in range(n_samples):
        a = grf_coeff(N, rng, lengthscale=lengthscale, sigma=sigma)
        A[k] = a
        U[k] = solve(assemble(a), f).reshape(N, N)
    return A, U
