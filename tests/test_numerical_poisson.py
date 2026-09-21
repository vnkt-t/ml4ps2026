"""Fast Poisson inverse uses the TPFA half-cell boundary convention exactly."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse.linalg as spla

from solver.assemble_fd import assemble
from solver.poisson import poisson_preconditioner, solve_poisson


@pytest.mark.parametrize("n", [1, 2, 7, 16, 32])
@pytest.mark.parametrize("coefficient", [0.125, 1.0, 500.0])
def test_fast_inverse_matches_sparse_direct_solve(n, coefficient):
    rng = np.random.default_rng(5)
    rhs = rng.normal(size=(n, n))
    A = assemble(np.full_like(rhs, coefficient))
    fast = solve_poisson(rhs, coefficient)
    direct = spla.spsolve(A, rhs.reshape(-1)).reshape(n, n)
    np.testing.assert_allclose(fast, direct, rtol=5e-12, atol=5e-16)
    np.testing.assert_allclose((A @ fast.reshape(-1)).reshape(n, n), rhs, rtol=2e-12, atol=2e-13)


@pytest.mark.parametrize("mode", [1, 8])
def test_extreme_sine_modes_and_normalization(mode):
    n = 8
    x = (np.arange(n) + 0.5) / n
    u = np.sin(np.pi * mode * x)[:, None] * np.sin(np.pi * mode * x)[None, :]
    eigenvalue = 8 * n*n * np.sin(np.pi * mode / (2*n))**2
    np.testing.assert_allclose(solve_poisson(eigenvalue * u), u, atol=1e-14)


def test_poisson_preconditioner_is_symmetric_positive_and_exact_for_constant_a():
    n = 9
    rng = np.random.default_rng(9)
    x, y = rng.normal(size=(2, n*n))
    M = poisson_preconditioner(n, coefficient=3.0)
    assert x @ (M @ x) > 0.0
    assert x @ (M @ y) == pytest.approx(y @ (M @ x), abs=1e-14)
    A = assemble(np.full((n, n), 3.0))
    corrected, info = spla.cg(A, x, M=M, maxiter=1, rtol=1e-12)
    np.testing.assert_allclose(A @ corrected, x, atol=1e-13)


def test_heterogeneous_preconditioned_spectrum_obeys_coefficient_bounds():
    # Facewise ellipticity, including Dirichlet walls, implies B <= A <= kappa B.
    # The generalized eigenvalue check is independent of the FFT implementation.
    n, kappa = 6, 500.0
    rng = np.random.default_rng(49)
    a = np.exp(rng.uniform(0.0, np.log(kappa), size=(n, n)))
    A, B = assemble(a).toarray(), assemble(np.ones_like(a)).toarray()
    spectrum = la.eigvalsh(A, B)
    assert spectrum.min() >= a.min() * (1.0 - 1e-12)
    assert spectrum.max() <= a.max() * (1.0 + 1e-12)
