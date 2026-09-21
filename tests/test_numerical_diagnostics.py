"""Independent finite-volume conservation and nonnegative energy checks."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from solver.assemble_fd import assemble, grid_centers
from solver.diagnostics import cell_energy, cell_flux_magnitude, face_fluxes
from solver.residual import residual
from solver.solve import solve


@pytest.mark.parametrize("n,contrast", [(1, 1.0), (8, 1.0), (12, 500.0)])
def test_face_flux_divergence_matches_operator(n, contrast):
    rng = np.random.default_rng(17)
    a = np.exp(rng.uniform(0.0, np.log(contrast), (n, n)))
    e = rng.normal(size=(n, n))
    qx, qy = face_fluxes(a, e)
    divergence = n * (np.diff(qx, axis=0) + np.diff(qy, axis=1))
    np.testing.assert_allclose(divergence.reshape(-1), assemble(a) @ e.reshape(-1), rtol=3e-14, atol=1e-10)


@pytest.mark.parametrize("n,contrast", [(1, 1.0), (8, 1.0), (12, 500.0)])
def test_cell_energy_partitions_quadratic_form(n, contrast):
    rng = np.random.default_rng(71)
    a = np.exp(rng.uniform(0.0, np.log(contrast), (n, n)))
    e = rng.normal(size=(n, n))
    A = assemble(a)
    energy = cell_energy(A, e)
    assert energy.shape == e.shape
    assert np.all(energy >= 0.0)
    np.testing.assert_allclose(energy.sum(), e.reshape(-1) @ (A @ e.reshape(-1)), rtol=2e-14)


def test_constant_field_has_only_boundary_energy():
    n = 7
    a = np.full((n, n), 3.0)
    e = np.ones_like(a)
    energy = cell_energy(assemble(a), e)
    assert np.count_nonzero(energy[1:-1, 1:-1]) == 0
    # 4*n boundary faces, each with transmissibility 2*a*n².
    assert energy.sum() == pytest.approx(4 * n * 2 * 3 * n**2)
    assert energy[0, 0] == pytest.approx(2 * energy[0, 1])


def test_flux_does_not_equal_flux_divergence():
    n = 9
    X, _ = grid_centers(n)
    a = np.ones_like(X)
    flux = cell_flux_magnitude(a, X)
    residual = (assemble(a) @ X.reshape(-1)).reshape(X.shape)
    np.testing.assert_allclose(flux[1:-1, 1:-1], 1.0, atol=1e-14)
    np.testing.assert_allclose(residual[1:-1, 1:-1], 0.0, atol=1e-13)


def test_energy_rejects_operator_without_diffusion_partition():
    with pytest.raises(ValueError, match="nonpositive"):
        cell_energy(sp.csr_matrix([[2.0, 1.0], [1.0, 2.0]]), np.ones(2))
    with pytest.raises(ValueError, match="symmetric"):
        cell_energy(sp.csr_matrix([[2.0, -1.0], [0.0, 2.0]]), np.ones(2))


def test_assembly_remains_finite_for_large_finite_coefficients():
    A = assemble(np.full((2, 2), 1e200))
    assert np.all(np.isfinite(A.data))
    np.testing.assert_allclose(A.toarray() / 1e200, assemble(np.ones((2, 2))).toarray())


@pytest.mark.parametrize("n", [0, -1, True, 2.5])
def test_invalid_grid_size_fails_early(n):
    with pytest.raises(ValueError):
        grid_centers(n)


def test_singular_solve_does_not_silently_return_nan():
    with pytest.raises(FloatingPointError, match="singular"):
        solve(sp.csr_matrix([[1.0, 1.0], [1.0, 1.0]]), np.ones(2))


def test_nonfinite_solver_inputs_fail_before_library_call():
    A = sp.eye(2, format="csr")
    with pytest.raises(ValueError, match="finite"):
        solve(A, [np.nan, 1.0])
    with pytest.raises(ValueError, match="finite"):
        residual(A, [np.inf, 1.0], np.ones(2))
