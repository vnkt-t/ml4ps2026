"""Bounded smooth coefficient fields with solved finite-volume targets.

Each draw smooths independent grid white noise with a periodic Gaussian filter
of standard deviation ``lengthscale * N`` cells, rescales the sampled minimum
and maximum to zero and one, then exponentiates into ``[a_min, a_min*kappa]``.
The normalization conditions the extrema separately for every field: this is
not an unmodified Gaussian or lognormal random-field law. The filter is applied
to the latent field, not to the exponentiated coefficient. Periodic filtering
does not change the solver's homogeneous Dirichlet boundary conditions.
"""
from __future__ import annotations

import operator

import numpy as np
from scipy.ndimage import gaussian_filter

from data.generate_contrast import default_rhs
from solver.assemble_fd import assemble
from solver.solve import solve


def smooth_field(
    N: int = 32,
    kappa: float = 100.0,
    rng: np.random.Generator | None = None,
    lengthscale: float = 0.12,
    a_min: float = 1.0,
) -> np.ndarray:
    """Return one continuous-valued coefficient array of shape ``(N, N)``.

    ``lengthscale`` is the filter standard deviation in units of the unit
    domain, not a claim about the final field's correlation length. Supplying
    a Generator makes draws reproducible and advances only that RNG's state.
    ``kappa=1`` gives a constant coefficient; otherwise a numerically degenerate
    smoothed draw raises an error rather than silently changing the family.
    """
    try:
        n = operator.index(N)
    except TypeError as exc:
        raise ValueError("N must be an integer >= 4") from exc
    if isinstance(N, (bool, np.bool_)) or n < 4:
        raise ValueError("N must be an integer >= 4")
    kappa, lengthscale, a_min = float(kappa), float(lengthscale), float(a_min)
    if not np.isfinite(kappa) or kappa < 1.0:
        raise ValueError("kappa must be finite and >= 1")
    if not np.isfinite(lengthscale) or lengthscale <= 0.0:
        raise ValueError("lengthscale must be finite and positive")
    if not np.isfinite(a_min) or a_min <= 0.0:
        raise ValueError("a_min must be finite and positive")
    a_max = a_min * kappa
    sigma = lengthscale * n
    if not np.isfinite(a_max) or not np.isfinite(sigma):
        raise ValueError("a_min*kappa and lengthscale*N must be finite")
    if rng is None:
        rng = np.random.default_rng()
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator or None")

    latent = gaussian_filter(rng.standard_normal((n, n)), sigma=sigma, mode="wrap")
    if kappa == 1.0:
        return np.full((n, n), a_min, dtype=np.float64)
    lo, hi = float(latent.min()), float(latent.max())
    span = hi - lo
    scale = max(abs(lo), abs(hi), np.finfo(np.float64).tiny)
    if not np.all(np.isfinite(latent)) or span <= 32 * np.finfo(np.float64).eps * scale:
        raise FloatingPointError("smoothed draw has no numerically usable variation")
    normalized = (latent - lo) / span
    coefficient = a_min * np.exp(np.log(kappa) * normalized)
    # Correct only endpoint roundoff; no thresholding or binary mask is used.
    coefficient = np.clip(coefficient, a_min, a_max)
    if not np.all(np.isfinite(coefficient)):
        raise FloatingPointError("coefficient transformation produced non-finite values")
    return coefficient


def smooth_sample(
    N: int = 32,
    kappa: float = 100.0,
    rng: np.random.Generator | None = None,
    lengthscale: float = 0.12,
    a_min: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(a, f, u_true)`` with the existing fixed forcing and solver."""
    a = smooth_field(N=N, kappa=kappa, rng=rng, lengthscale=lengthscale, a_min=a_min)
    f = default_rhs(a.shape[0])
    u_true = solve(assemble(a), f).reshape(a.shape)
    return a, f, u_true
