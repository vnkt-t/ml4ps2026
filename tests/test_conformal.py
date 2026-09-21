"""Regression checks for interval validity, dependence, and frozen scale rules."""
import itertools

import numpy as np
import pytest

from uncertainty.conformal import (
    ConformalCalibration,
    calibrate_cellwise,
    calibrate_fieldwise,
    conformal_quantile,
    coverage_metrics,
    interval_radius,
    safe_scale,
)
from uncertainty.weighted_conformal import (
    calibrate_weighted,
    coverage_from_scores,
    density_ratio_from_classifier,
    effective_sample_size,
    fit_density_ratio_classifier,
    normalize_density_ratio_weights,
    weighted_conformal_quantile,
)


def test_finite_sample_rank_and_infinite_mass_are_preserved():
    assert conformal_quantile(np.arange(1.0, 10.0), 0.1) == 9.0
    assert np.isinf(conformal_quantile(np.arange(1.0, 9.0), 0.1))
    # Dropping the +inf observation would incorrectly give threshold 3.
    assert np.isinf(conformal_quantile([1.0, 2.0, 3.0, np.inf], 0.25))
    assert np.isinf(weighted_conformal_quantile([1, 2, 3, np.inf], np.ones(4), 0.25))


@pytest.mark.parametrize("scores", [[], [1.0, np.nan], [1.0, -np.inf]])
def test_invalid_scores_fail_instead_of_changing_calibration_population(scores):
    with pytest.raises(ValueError):
        conformal_quantile(scores, 0.1)
    with pytest.raises(ValueError):
        weighted_conformal_quantile(scores, np.ones(len(scores)), 0.1)
    with pytest.raises(ValueError):
        coverage_from_scores(scores, 1.0)


@pytest.mark.parametrize("weights", [[1.0, np.nan], [1.0, np.inf], [1.0, -1.0]])
def test_invalid_weights_are_not_silently_imputed(weights):
    with pytest.raises(ValueError):
        normalize_density_ratio_weights(weights)
    with pytest.raises(ValueError):
        weighted_conformal_quantile([1.0, 2.0], weights, 0.1)


@pytest.mark.parametrize("test_weight", [-1.0, np.nan, np.inf])
def test_invalid_test_mass_rejected(test_weight):
    with pytest.raises(ValueError):
        weighted_conformal_quantile([1.0, 2.0], [1.0, 1.0], 0.1, test_weight=test_weight)


def test_weighted_quantile_matches_unweighted_when_ratios_equal():
    rng = np.random.default_rng(82)
    for n in [2, 8, 9, 20, 99]:
        scores = rng.normal(size=n)
        for alpha in [0.1, 0.2, 0.5]:
            assert weighted_conformal_quantile(scores, np.ones(n), alpha) == conformal_quantile(scores, alpha)


def test_weighted_quantile_requires_individual_target_mass_and_common_scaling():
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.array([0.5, 1.0, 1.5, 2.0])
    small = weighted_conformal_quantile(scores, weights, 0.25, test_weight=0.1)
    large = weighted_conformal_quantile(scores, weights, 0.25, test_weight=10.0)
    assert small == 4.0
    assert np.isinf(large)
    assert weighted_conformal_quantile(scores, weights * 13.0, 0.25, test_weight=1.3) == small
    # Normalization must not overflow for very large but finite true ratios.
    assert weighted_conformal_quantile(scores, np.full(4, 1e308), 0.25, test_weight=1e308) == 4.0
    assert effective_sample_size(np.full(4, 1e308)) == 4.0


def test_exact_weighted_rule_covers_under_enumerated_covariate_shift():
    # Exhaustive finite probability space, independent calibration fields from
    # P and a test field from Q. No Monte Carlo tolerance hides rank errors.
    source = np.array([1 / 3, 1 / 3, 1 / 3])
    target = np.array([0.1, 0.3, 0.6])
    ratio = target / source
    for alpha in [0.1, 0.25, 0.5]:
        coverage = 0.0
        for states in itertools.product(range(3), repeat=4):
            calibration = np.array(states[:3])
            test = states[3]
            qhat = weighted_conformal_quantile(calibration, ratio[calibration], alpha, test_weight=ratio[test])
            probability = float(np.prod(source[calibration]) * target[test])
            coverage += probability * (test <= qhat)
        assert coverage >= 1.0 - alpha - 1e-12


def test_zero_calibration_mass_retains_target_mass_at_infinity():
    assert np.isinf(weighted_conformal_quantile([1, 2], [0, 0], 0.1, test_weight=1))
    with pytest.raises(ValueError):
        weighted_conformal_quantile([1, 2], [0, 0], 0.1, test_weight=0)


def test_frozen_scale_floor_makes_prediction_independent_of_target_batch():
    error = np.full((10, 2, 2), 0.2)
    scale = np.ones_like(error)
    calibration = calibrate_cellwise(error, scale=scale)
    target = np.zeros((1, 2, 2))
    together = np.concatenate([target, np.full((1, 2, 2), 1e8)])
    assert calibration.scale_floor == 0.05
    np.testing.assert_array_equal(interval_radius(calibration, target), interval_radius(calibration, together)[:1])
    with pytest.raises(ValueError, match="requires a scale"):
        interval_radius(calibration)


def test_fieldwise_calibration_counts_independent_fields_not_cells():
    errors = np.arange(1.0, 9.0).reshape(8, 1, 1) * np.ones((8, 3, 3))
    field = calibrate_fieldwise(errors, alpha=0.1)
    cells = calibrate_cellwise(errors, alpha=0.1)
    assert field.n_calibration == 8
    assert cells.n_calibration == 72
    assert np.isinf(field.qhat)
    assert np.isfinite(cells.qhat)
    metrics = coverage_metrics(errors, field)
    assert metrics["fieldwise_coverage"] == 1.0
    assert np.isinf(metrics["mean_width"])


def test_fieldwise_scale_floor_is_fixed_independently_of_calibration():
    error = np.ones((10, 2, 2))
    scale = np.ones_like(error)
    calibration = calibrate_fieldwise(error, scale=scale)
    assert calibration.scale_floor == 1e-12
    metrics = coverage_metrics(error, calibration, scale=scale)
    assert metrics["fieldwise_coverage"] == 1.0


@pytest.mark.parametrize("invalid", [[-1.0, 1.0], [np.nan, 1.0], [np.inf, 1.0]])
def test_nonnegative_error_and_scale_contracts(invalid):
    with pytest.raises(ValueError):
        calibrate_cellwise(invalid)
    with pytest.raises(ValueError):
        safe_scale(invalid)


def test_scale_shape_mismatch_cannot_silently_add_a_sample_axis():
    with pytest.raises(ValueError, match="cannot broadcast"):
        calibrate_cellwise(np.ones((3, 4)), scale=np.ones((3, 1, 4)))
    with pytest.raises(ValueError, match="cannot broadcast"):
        coverage_metrics(np.ones((3, 4)), ConformalCalibration("bad", 0.1, 1.0, "cellwise"), np.ones((3, 1, 4)))


def test_density_ratio_source_and_target_keep_one_common_normalization():
    rng = np.random.default_rng(21)
    source = rng.normal(size=(80, 2))
    target = rng.normal(loc=1.4, size=(40, 2))
    fitted = fit_density_ratio_classifier(source, target, random_state=21)
    source_raw = density_ratio_from_classifier(fitted.estimator, source, n_source=80, n_target=40)
    target_raw = density_ratio_from_classifier(fitted.estimator, target, n_source=80, n_target=40)
    np.testing.assert_allclose(fitted.source_weights, source_raw / source_raw.mean())
    np.testing.assert_allclose(fitted.target_weights, target_raw / source_raw.mean())
    assert not np.isclose(fitted.target_weights.mean(), 1.0)
