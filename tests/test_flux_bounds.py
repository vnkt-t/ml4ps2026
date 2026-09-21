"""Independent energy, incidence, and sparse-inverse checks of flux bounds."""
import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from solver.assemble_fd import assemble
from uncertainty.flux_bounds import equilibrated_flux_energy, poisson_flux_bounds
from uncertainty.rank_bounds import magnitude_envelope, poisson_error_bound, separate_top_set


def _incidence_and_weights(a):
    n = a.shape[0]
    columns, unit_weights, variable_weights = [], [], []
    for i in range(n):
        for j in range(n):
            origin = i * n + j
            for di, dj in [(1, 0), (0, 1)]:
                if i + di < n and j + dj < n:
                    column = np.zeros(n * n)
                    column[origin] = 1.0
                    column[(i + di) * n + j + dj] = -1.0
                    face = 2 * a[i, j] * a[i + di, j + dj] / (a[i, j] + a[i + di, j + dj])
                    columns.append(column); unit_weights.append(n*n); variable_weights.append(n*n*face)
            for wall in [i == 0, i == n-1, j == 0, j == n-1]:
                if wall:
                    column = np.zeros(n * n); column[origin] = 1.0
                    columns.append(column); unit_weights.append(2*n*n); variable_weights.append(2*n*n*a[i, j])
    return np.stack(columns, axis=1), np.array(unit_weights), np.array(variable_weights)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_flux_energy_matches_independent_graph_construction(n):
    rng = np.random.default_rng(41+n)
    a = np.exp(rng.uniform(-2, 3, size=(n, n)))
    d = rng.normal(size=(n, n))
    result = poisson_flux_bounds(d, a)
    D, wb, wa = _incidence_and_weights(a)
    np.testing.assert_allclose(np.einsum("if,f,jf->ij", D, wa, D, optimize=False),
                               assemble(a).toarray(), rtol=2e-14, atol=1e-13)
    q = wb * np.einsum("if,i->f", D, result.poisson_potential.reshape(-1), optimize=False)
    np.testing.assert_allclose(np.einsum("if,f->i", D, q, optimize=False), d.reshape(-1), rtol=2e-13, atol=2e-13)
    assert result.equilibrated_energy == pytest.approx(float(np.sum(q*q/wa)), rel=2e-14)


@pytest.mark.parametrize("n", [1, 3, 8])
@pytest.mark.parametrize("coefficient", [0.02, 1.0, 500.0])
def test_constant_coefficient_flux_energy_is_exact_and_bound_matches_original(n, coefficient):
    rng = np.random.default_rng(n)
    a = np.full((n, n), coefficient)
    d = rng.normal(size=(n, n))
    result = poisson_flux_bounds(d, a)
    exact = spsolve(assemble(a), d.reshape(-1)).reshape(n, n)
    assert result.equilibrated_energy == pytest.approx(float(np.sum(d * exact)), rel=3e-14)
    np.testing.assert_allclose(result.bound, result.baseline_bound, rtol=3e-14, atol=1e-16)
    np.testing.assert_allclose(result.bound, poisson_error_bound(d, coefficient), rtol=3e-14, atol=1e-16)


@pytest.mark.parametrize("contrast", [3.0, 100.0, 5000.0])
def test_variable_coefficient_majorant_is_valid_and_no_looser_than_original(contrast):
    rng = np.random.default_rng(31)
    a = 0.03 * np.exp(rng.uniform(0, np.log(contrast), size=(7, 7)))
    d = rng.normal(size=(7, 7))
    result = poisson_flux_bounds(d, a)
    exact = spsolve(assemble(a), d.reshape(-1)).reshape(7, 7)
    assert float(np.sum(d * exact)) <= result.equilibrated_energy * (1 + 1e-12)
    assert result.equilibrated_energy <= result.poisson_energy / a.min() * (1 + 1e-12)
    assert np.all(np.abs(exact) <= result.bound * (1 + 1e-12))
    assert np.all(result.bound <= result.baseline_bound * (1 + 1e-12))
    assert np.any(result.bound < result.baseline_bound * 0.99)
    correction = rng.normal(scale=0.02, size=(7, 7))
    error = correction + exact
    lower, upper = magnitude_envelope(correction, result.bound)
    high, low = separate_top_set(lower, upper, .1)
    n_top = int(np.ceil(.1 * error.size)); truth_top = np.zeros(error.size, dtype=bool)
    truth_top[np.argpartition(np.abs(error).reshape(-1), -n_top)[-n_top:]] = True
    assert not np.any(high.reshape(-1) & ~truth_top)
    assert not np.any(low.reshape(-1) & truth_top)


def test_constant_green_column_attains_the_pointwise_bound_at_source():
    a = np.full((5, 5), 0.3); defect = np.zeros((5, 5)); defect[2, 2] = 1
    result = poisson_flux_bounds(defect, a)
    exact = spsolve(assemble(a), defect.reshape(-1)).reshape(5, 5)
    assert result.bound[2, 2] == pytest.approx(abs(exact[2, 2]), rel=2e-14)


def test_zero_defect_and_invalid_input_contracts():
    result = poisson_flux_bounds(np.zeros((3, 3)), np.ones((3, 3)))
    assert result.equilibrated_energy == 0
    np.testing.assert_array_equal(result.bound, np.zeros((3, 3)))
    with pytest.raises(ValueError):
        poisson_flux_bounds(np.ones((3, 2)), np.ones((3, 2)))
    with pytest.raises(ValueError):
        poisson_flux_bounds(np.ones((3, 3)), np.zeros((3, 3)))
    with pytest.raises(ValueError):
        equilibrated_flux_energy(np.ones((3, 3)), np.full((3, 3), np.nan))
