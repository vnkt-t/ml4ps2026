"""Independent checks of pointwise error bounds and exact interval separation."""
import itertools

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from solver.assemble_fd import assemble
from uncertainty.rank_bounds import (
    poisson_inverse_diagonal, poisson_error_bound, magnitude_envelope, separate_top_set,
)


@pytest.mark.parametrize("n", [1, 2, 5, 8])
def test_poisson_diagonal_agrees_with_dense_inverse(n):
    B = assemble(np.ones((n, n))).toarray()
    expected = np.diag(np.linalg.inv(B)).reshape(n, n)
    np.testing.assert_allclose(poisson_inverse_diagonal(n), expected, rtol=2e-14, atol=1e-16)


@pytest.mark.parametrize("contrast", [1, 10, 500])
@pytest.mark.parametrize("minimum", [0.1, 1.0, 7.0])
def test_pointwise_bound_covers_independent_direct_correction(contrast, minimum):
    rng = np.random.default_rng(51)
    n = 7
    a = minimum * np.where(rng.random((n, n)) < .45, contrast, 1.0)
    A = assemble(a)
    residual = rng.normal(size=n*n)
    exact = spsolve(A, residual).reshape(n, n)
    correction = rng.normal(size=(n, n)) * 0.01
    defect = (residual - A @ correction.reshape(-1)).reshape(n, n)
    bound = poisson_error_bound(defect, minimum)
    assert np.all(np.abs(exact - correction) <= bound * (1 + 1e-12))
    lower, upper = magnitude_envelope(correction, bound)
    assert np.all(lower <= np.abs(exact) + 1e-12)
    assert np.all(upper >= np.abs(exact) - 1e-12)


def test_bound_is_attained_for_constant_coefficient_green_column():
    n, minimum, index = 5, .3, 12
    A = assemble(np.full((n, n), minimum))
    defect = np.zeros(n*n)
    defect[index] = 1
    exact = spsolve(A, defect)
    bound = poisson_error_bound(defect.reshape(n, n), minimum).reshape(-1)
    assert bound[index] == pytest.approx(abs(exact[index]), rel=2e-14)


def test_strict_top_set_flags_hold_for_all_enclosure_corners():
    lower = np.array([.9, .7, .3, .15, .0])
    upper = np.array([1., .8, .75, .2, .1])
    high, low = separate_top_set(lower, upper, .4)
    assert high[0] and low[3] and low[4]
    assert not high[1] and not low[1]  # overlapping rank enclosure
    for corner in itertools.product([0, 1], repeat=5):
        truth = np.where(corner, upper, lower)
        top = np.zeros(5, dtype=bool)
        top[np.argsort(truth)[-2:]] = True
        assert not np.any(high & ~top)
        assert not np.any(low & top)


def test_equal_intervals_do_not_certify_ambiguous_ties():
    high, low = separate_top_set(np.ones(5), np.ones(5), .4)
    assert not high.any() and not low.any()


def test_invalid_bound_inputs_fail():
    with pytest.raises(ValueError):
        poisson_error_bound(np.ones((3, 3)), 0)
    with pytest.raises(ValueError):
        magnitude_envelope(np.zeros(3), -np.ones(3))
    with pytest.raises(ValueError):
        separate_top_set(np.ones(3), np.zeros(3))
