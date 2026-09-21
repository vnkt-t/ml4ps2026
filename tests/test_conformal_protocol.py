"""Protocol regression checks for the saved-bundle OOD diagnostics."""
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from experiments.exp9_real_fno_ood_conformal import (
    _bootstrap_masked_coverage,
    _calibrate_methods,
    _hybrid_scale,
    width_localization_metrics,
)
from experiments.exp10_ood_abstention import _run_shift


def test_constant_width_has_chance_auroc():
    result = width_localization_metrics(np.ones((4, 4)), np.arange(16.0).reshape(4, 4))
    assert result == {"spearman": 0.0, "auroc_top10": 0.5}


def test_masked_bootstrap_estimates_reported_pooled_cell_coverage():
    # Equal field weights would produce .5, but there is 1 covered cell among
    # 5 masked cells. The confidence interval must target the latter ratio.
    covered = np.array([[True, False, False, False], [False, False, False, False]])
    mask = np.array([[True, False, False, False], [True, True, True, True]])
    stats = _bootstrap_masked_coverage(covered, mask, np.random.default_rng(0), 0)
    assert stats == {"mean": 0.2, "ci95_low": 0.2, "ci95_high": 0.2}


def test_hybrid_scale_does_not_refit_floors_to_target_batch():
    calibration = {name: np.ones((10, 2, 2)) for name in ["abs_e", "sigma_ens", "abs_r", "jacobi20", "rung4"]}
    _, medians = _calibrate_methods(calibration, 0.1)
    one = {name: np.zeros((1, 2, 2)) for name in ["sigma_ens", "rung4"]}
    batch = {name: np.concatenate([value, np.full_like(value, 1e8)]) for name, value in one.items()}
    np.testing.assert_array_equal(_hybrid_scale(one, medians), _hybrid_scale(batch, medians)[:1])


def _args():
    return Namespace(seed=31, train_fraction=0.6, high_error_quantile=0.85,
                     risk_model="ridge", ridge_alpha=1.0, bootstrap=0,
                     operating_rates=[0.1, 0.5])


def _features():
    value = np.linspace(0.01, 0.9, 30)
    return pd.DataFrame({"rel_l2_error": value, "ens_sigma_mean": value + 0.01,
                         "res_abs_mean": value ** 2, "ladder_rung4_mean": value * 2})


def test_within_shift_training_does_not_use_heldout_error_labels():
    df = _features()
    args = _args()
    baseline = _run_shift("test_id", df, args, np.random.default_rng(1), 0)
    changed = df.copy()
    changed.loc[baseline["heldout_indices"], "rel_l2_error"] *= 100.0
    rerun = _run_shift("test_id", changed, args, np.random.default_rng(1), 0)
    assert baseline["high_error_threshold"] == rerun["high_error_threshold"]
    assert baseline["heldout_risk_scores"] == rerun["heldout_risk_scores"]
    assert baseline["heldout_high_error_labels"] != rerun["heldout_high_error_labels"]
    assert set(baseline["training_indices"]).isdisjoint(baseline["heldout_indices"])
    assert baseline["target_labels_used_for_risk_training"] is True
    assert baseline["solver_handoff_executed"] is False
    assert baseline["handoff_unit"] == "whole_field"


def test_empty_risk_training_or_test_split_fails_clearly():
    args = _args()
    args.train_fraction = 1.0
    with pytest.raises(ValueError, match="train_fraction"):
        _run_shift("test_id", _features(), args, np.random.default_rng(1), 0)


def test_tied_risk_scores_have_row_order_invariant_expected_handoff_curve():
    from analysis.audit_ood_protocol import expected_rank_curve
    scores = np.array([2.0, 1.0, 1.0, 1.0])
    high = np.array([False, True, True, False])
    curve = expected_rank_curve(scores, high)
    np.testing.assert_allclose(curve, [0.5, 0.5, 1 / 3, 1 / 6, 0.0], atol=1e-15)
    permutation = np.array([3, 0, 2, 1])
    np.testing.assert_allclose(curve, expected_rank_curve(scores[permutation], high[permutation]))


def test_id_transfer_never_fits_or_sets_threshold_using_target_error_labels():
    from analysis.audit_ood_protocol import id_transfer
    tables = {name: _features() for name in ["test_id", "test_r", "test_b", "test_cs"]}
    args = _args()
    args.bootstrap_seed = 12
    baseline = id_transfer(tables, args)
    changed = {name: value.copy() for name, value in tables.items()}
    for name in ["test_r", "test_b", "test_cs"]:
        changed[name]["rel_l2_error"] *= 100.0
    changed["test_id"].loc[baseline["id_heldout_indices"], "rel_l2_error"] *= 100.0
    rerun = id_transfer(changed, args)
    assert baseline["threshold"] == rerun["threshold"]
    assert baseline["training_indices"] == rerun["training_indices"]
    for dataset in tables:
        for method in ["ensemble_only", "combined"]:
            np.testing.assert_array_equal(baseline["per_shift"][dataset]["risk_scores"][method],
                                          rerun["per_shift"][dataset]["risk_scores"][method])
        assert baseline["per_shift"][dataset]["risk_target_labels_used_for_training"] is False


def test_cheap_feature_controls_keep_target_errors_out_of_training():
    from analysis.check_id_risk_controls import evaluate_controls
    tables = {}
    for name in ["test_id", "test_r", "test_b", "test_cs"]:
        table = _features()
        table["prediction_magnitude_mean"] = np.linspace(0.2, 0.8, len(table))
        table["coefficient_log_a_mean"] = np.linspace(-0.5, 0.5, len(table))
        tables[name] = table
    baseline = evaluate_controls(tables, seed=31, bootstrap=0)
    changed = {name: table.copy() for name, table in tables.items()}
    for name in ["test_r", "test_b", "test_cs"]:
        changed[name]["rel_l2_error"] *= 100
    rerun = evaluate_controls(changed, seed=31, bootstrap=0)
    assert baseline["threshold"] == rerun["threshold"]
    for columns in baseline["feature_sets"].values():
        assert not any("error" in column for column in columns)
    for dataset in tables:
        for method in baseline["feature_sets"]:
            np.testing.assert_array_equal(baseline["per_shift"][dataset]["risk_scores"][method],
                                          rerun["per_shift"][dataset]["risk_scores"][method])
