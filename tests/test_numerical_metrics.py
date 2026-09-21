"""Localization metrics agree with standard definitions and unbiased ties."""
from __future__ import annotations

import itertools

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from analysis.metrics import compute_metrics, localization_metrics_top10


def test_constant_score_has_chance_overlap_and_capture_without_warnings():
    e = np.arange(1.0, 101.0).reshape(10, 10)
    m = compute_metrics(np.ones_like(e), e)
    assert m["spearman"] == 0.0
    assert m["auroc"] == pytest.approx(0.5)
    assert m["error_capture"] == pytest.approx(0.2)
    for q in (10, 20, 30):
        assert m[f"top{q}_overlap"] == pytest.approx(q / 100.0)


def test_top10_constant_score_is_chance_and_flattened_safe():
    e = np.arange(1.0, 101.0)
    m = localization_metrics_top10(np.ones(100), e)
    assert m == pytest.approx({"spearman": 0.0, "auroc": 0.5,
                               "error_capture_top10": 0.1, "top10_overlap": 0.1})
    assert m == localization_metrics_top10(np.ones((10, 10)), e.reshape(10, 10))


def test_top10_geometry_ties_are_invariant_to_grid_order():
    n = 10
    x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    boundary_distance = np.minimum.reduce([x, y, n-1-x, n-1-y]).reshape(-1).astype(float)
    e = np.arange(1.0, n*n + 1.0)
    rng = np.random.default_rng(37)
    permutation = rng.permutation(e.size)
    expected = localization_metrics_top10(boundary_distance, e)
    actual = localization_metrics_top10(boundary_distance[permutation], e[permutation])
    assert actual == pytest.approx(expected)


def test_top10_untied_oracle_and_standard_auc():
    rng = np.random.default_rng(101)
    e = rng.random(100)
    m = localization_metrics_top10(e, e)
    assert m["spearman"] == pytest.approx(1.0)
    assert m["auroc"] == pytest.approx(1.0)
    assert m["top10_overlap"] == pytest.approx(1.0)
    s = rng.random(100)
    y = np.zeros(e.size)
    y[np.argsort(e)[-10:]] = 1
    assert localization_metrics_top10(s, e)["auroc"] == pytest.approx(roc_auc_score(y, s))


def test_untied_metrics_match_original_definitions():
    rng = np.random.default_rng(4)
    s, e = rng.random((2, 10, 10))
    m = compute_metrics(s, e)
    y = np.zeros(e.size)
    y[np.argsort(e.reshape(-1))[-20:]] = 1
    assert m["auroc"] == pytest.approx(roc_auc_score(y, s.reshape(-1)))
    for q in (10, 20, 30):
        top_s = set(np.argsort(s.reshape(-1))[-q:])
        top_e = set(np.argsort(e.reshape(-1))[-q:])
        assert m[f"top{q}_overlap"] == len(top_s & top_e) / q


def test_error_cutoff_tie_auc_equals_exhaustive_random_tie_breaking():
    s = np.arange(1.0, 11.0)
    e = np.array([1, 2, 3, 4, 5, 6, 7, 9, 9, 9], dtype=float)
    values = []
    for positives in itertools.combinations([7, 8, 9], 2):
        y = np.zeros(10)
        y[list(positives)] = 1
        values.append(roc_auc_score(y, s))
    m = compute_metrics(s.reshape(2, 5), e.reshape(2, 5))
    assert m["auroc"] == pytest.approx(np.mean(values))


def test_tied_metrics_are_permutation_invariant():
    rng = np.random.default_rng(52)
    s, e = rng.integers(0, 4, size=(2, 100)).astype(float)
    perm = rng.permutation(100)
    left = compute_metrics(s.reshape(10, 10), e.reshape(10, 10))
    right = compute_metrics(s[perm].reshape(10, 10), e[perm].reshape(10, 10))
    for key in left.keys() - {"com_dist"}:
        assert left[key] == pytest.approx(right[key])


@pytest.mark.parametrize("score,error", [
    (np.ones((2, 2)), np.ones((1, 4))),
    (np.empty((0, 0)), np.empty((0, 0))),
    (np.ones((2, 2)), -np.ones((2, 2))),
    (np.ones((2, 2)), np.full((2, 2), np.nan)),
])
def test_invalid_metric_inputs_fail_explicitly(score, error):
    with pytest.raises(ValueError):
        compute_metrics(score, error)
