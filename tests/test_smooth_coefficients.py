"""Distribution, reproducibility and target consistency for smooth media."""
import numpy as np
import pytest

from data.generate_contrast import default_rhs
from data.generate_smooth import smooth_field, smooth_sample
from solver.assemble_fd import assemble


@pytest.mark.parametrize("N", [4, 17, 32])
def test_prescribed_range_and_continuous_values(N):
    a = smooth_field(N=N, kappa=100, a_min=2.5, rng=np.random.default_rng(31))
    assert a.shape == (N, N) and a.dtype == np.float64
    assert np.isfinite(a).all()
    assert a.min() >= 2.5 and a.max() <= 250
    np.testing.assert_allclose([a.min(), a.max()], [2.5, 250], rtol=1e-14)
    assert np.unique(a).size > 0.95 * a.size
    assert np.count_nonzero((a > 2.5) & (a < 250)) >= a.size - 2


def test_default_family_has_smooth_local_changes_without_binary_interfaces():
    a = smooth_field(rng=np.random.default_rng(902))
    log_a = np.log(a)
    adjacent_log_jumps = np.concatenate([
        (log_a - np.roll(log_a, 1, axis=axis)).ravel() for axis in (0, 1)
    ])
    # A binary interface would have a local ratio of the full contrast, 100.
    assert np.exp(np.abs(adjacent_log_jumps).max()) < 10
    # High-frequency variation is suppressed in the log coefficient, including
    # across the periodic filter's edge (the PDE itself is not periodic).
    spectrum = np.abs(np.fft.fft2(log_a - log_a.mean())) ** 2
    freq = np.fft.fftfreq(a.shape[0])
    high_frequency = (np.abs(freq[:, None]) >= 0.25) | (np.abs(freq[None, :]) >= 0.25)
    assert spectrum[high_frequency].sum() / spectrum.sum() < 1e-6


def test_reproducible_draws_advance_only_the_supplied_rng():
    first, replay = np.random.default_rng(91), np.random.default_rng(91)
    independent = np.random.default_rng(812)
    a1, a2 = smooth_field(rng=first), smooth_field(rng=first)
    np.testing.assert_array_equal(a1, smooth_field(rng=replay))
    smooth_field(rng=independent)
    np.testing.assert_array_equal(a2, smooth_field(rng=replay))
    assert not np.array_equal(a1, a2)
    assert not np.array_equal(a1, smooth_field(rng=np.random.default_rng(92)))


def test_sample_uses_same_fixed_forcing_and_discrete_solve():
    a, f, u = smooth_sample(N=16, rng=np.random.default_rng(101))
    np.testing.assert_array_equal(f, default_rhs(16))
    np.testing.assert_array_equal(a, smooth_field(N=16, rng=np.random.default_rng(101)))
    _, another_f, _ = smooth_sample(N=16, rng=np.random.default_rng(102))
    np.testing.assert_array_equal(f, another_f)
    assert u.shape == a.shape and np.isfinite(u).all()
    residual = assemble(a) @ u.ravel() - f.ravel()
    assert np.linalg.norm(residual) / np.linalg.norm(f) < 1e-11


def test_unit_contrast_is_constant():
    a = smooth_field(N=4, kappa=1, a_min=3, rng=np.random.default_rng(2))
    np.testing.assert_array_equal(a, np.full((4, 4), 3.0))


@pytest.mark.parametrize("kwargs", [
    {"N": 3}, {"N": 8.5}, {"N": True},
    {"kappa": 0.9}, {"kappa": np.inf}, {"kappa": np.nan},
    {"lengthscale": 0}, {"lengthscale": -0.1}, {"lengthscale": np.inf},
    {"lengthscale": np.nan}, {"a_min": 0}, {"a_min": -1},
    {"a_min": np.inf}, {"a_min": np.nan}, {"a_min": 1e308, "kappa": 100},
])
def test_invalid_parameters_fail_before_drawing(kwargs):
    rng, untouched = np.random.default_rng(1), np.random.default_rng(1)
    with pytest.raises(ValueError):
        smooth_field(rng=rng, **kwargs)
    assert rng.random() == untouched.random()


def test_degenerate_smoothed_field_fails_loudly(monkeypatch):
    monkeypatch.setattr("data.generate_smooth.gaussian_filter", lambda x, **kwargs: np.ones_like(x))
    with pytest.raises(FloatingPointError, match="no numerically usable variation"):
        smooth_field(rng=np.random.default_rng(1))
