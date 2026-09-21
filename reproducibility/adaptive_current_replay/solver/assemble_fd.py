"""Finite-volume (TPFA) assembly of the discrete operator for -div(a grad u) = f.

Cell-centered two-point flux approximation on an N x N grid over [0,1]^2 with
homogeneous Dirichlet boundary conditions and harmonic averaging of the coefficient
at cell faces (the standard choice for heterogeneous / high-contrast diffusion).

The SAME assembled A(a) returned here MUST be reused for:
  - the ground-truth solve            (solver/solve.py),
  - the residual r = A u_hat - b       (solver/residual.py),
  - the partial-inverse compute ladder (solver/relaxation.py, solver/multigrid.py),
  - the full-inverse oracle.
That consistency is what makes the discrete error identity  e = A^{-1} r  exact.

Conventions
-----------
- Grid: cell centers at ((i+0.5)h, (j+0.5)h), i,j in 0..N-1, h = 1/N.
- Coefficient a is given at cell centers, shape (N, N), strictly positive.
- Unknown ordering is row-major: vector index = i*N + j for cell (i, j).
- A is scaled by 1/h^2 so it approximates the continuous operator and the
  right-hand side is the pointwise source f sampled at cell centers (b = f).
- Transmissibility: interior face = harmonic mean of the two cell coeffs;
  boundary face = 2*a_cell (Dirichlet wall is h/2 from the cell center).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _harmonic(a: float, b: float) -> float:
    # Avoid overflowing a*b for finite, large coefficients.
    lo, hi = min(a, b), max(a, b)
    return lo * (2.0 / (1.0 + lo / hi))


def assemble(a: np.ndarray) -> sp.csr_matrix:
    """Assemble the (N^2 x N^2) symmetric positive-definite system matrix A(a).

    Parameters
    ----------
    a : (N, N) array of strictly positive cell-centered coefficients.

    Returns
    -------
    A : scipy.sparse.csr_matrix of shape (N*N, N*N), SPD, M-matrix.
    """
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.size == 0:
        raise ValueError(f"a must be square (N, N); got {a.shape}")
    if not np.all(np.isfinite(a)) or not np.all(a > 0):
        raise ValueError("coefficient field a must be finite and strictly positive")

    N = a.shape[0]
    h2 = (1.0 / N) ** 2

    def idx(i: int, j: int) -> int:
        return i * N + j

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    diag = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        for j in range(N):
            # +x neighbor
            if i + 1 < N:
                T = _harmonic(a[i, j], a[i + 1, j])
                rows.append(idx(i, j)); cols.append(idx(i + 1, j)); vals.append(-T / h2)
                diag[i, j] += T
            else:
                diag[i, j] += 2.0 * a[i, j]          # Dirichlet wall at distance h/2
            # -x neighbor
            if i - 1 >= 0:
                T = _harmonic(a[i, j], a[i - 1, j])
                rows.append(idx(i, j)); cols.append(idx(i - 1, j)); vals.append(-T / h2)
                diag[i, j] += T
            else:
                diag[i, j] += 2.0 * a[i, j]
            # +y neighbor
            if j + 1 < N:
                T = _harmonic(a[i, j], a[i, j + 1])
                rows.append(idx(i, j)); cols.append(idx(i, j + 1)); vals.append(-T / h2)
                diag[i, j] += T
            else:
                diag[i, j] += 2.0 * a[i, j]
            # -y neighbor
            if j - 1 >= 0:
                T = _harmonic(a[i, j], a[i, j - 1])
                rows.append(idx(i, j)); cols.append(idx(i, j - 1)); vals.append(-T / h2)
                diag[i, j] += T
            else:
                diag[i, j] += 2.0 * a[i, j]

    for i in range(N):
        for j in range(N):
            rows.append(idx(i, j)); cols.append(idx(i, j)); vals.append(diag[i, j] / h2)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(N * N, N * N))
    A._neuralops_coeff = a.copy()
    A._neuralops_grid_shape = (N, N)
    return A


def grid_centers(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, Y) meshgrids of cell-center coordinates, shape (N, N) each."""
    if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or N < 1:
        raise ValueError("N must be a positive integer")
    h = 1.0 / N
    c = (np.arange(N) + 0.5) * h
    X, Y = np.meshgrid(c, c, indexing="ij")
    return X, Y
