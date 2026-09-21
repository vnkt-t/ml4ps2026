"""Independent regressions for the frozen-field OOD certificate evaluator."""
import numpy as np
import pandas as pd
import pytest
from scipy.sparse.linalg import cg, spsolve

import experiments.exp_rank_bounds_ood as experiment
from solver.assemble_fd import assemble
from solver.poisson import poisson_preconditioner


def _field(seed=12):
    rng = np.random.default_rng(seed)
    coefficient = np.where(rng.uniform(size=(8, 8)) < 0.5, 0.02, 2.0)
    forcing = np.ones((8, 8))
    truth = spsolve(assemble(coefficient), forcing.reshape(-1)).reshape(8, 8)
    prediction = truth + 0.01 * rng.normal(size=(8, 8))
    return coefficient, forcing, prediction, truth


def test_actual_coefficient_minimum_and_signed_correction_are_used():
    coefficient, forcing, prediction, truth = _field()
    rows = experiment.evaluate_field(coefficient, forcing, prediction, truth, budgets=[1, 4])
    A = assemble(coefficient)
    residual = A @ prediction.reshape(-1) - forcing.reshape(-1)
    correction, _ = cg(A, residual, M=poisson_preconditioner(8), x0=np.zeros(64),
                        rtol=8 * np.finfo(float).eps, atol=0.0, maxiter=4)
    last = rows[-1]
    expected = np.linalg.norm(prediction - correction.reshape(8, 8) - truth) / np.linalg.norm(truth)
    assert last["corrected_prediction_relative_l2"] == pytest.approx(expected, rel=1e-13)
    assert last["a_min"] == 0.02
    assert last["actual_coefficient_contrast"] == 100.0
    for row in rows[2:]:
        assert row["false_high_count"] == row["false_low_count"] == 0
        assert row["bound_violations_beyond_numeric_tolerance"] == 0
        assert row["magnitude_violations_beyond_numeric_tolerance"] == 0
        assert 0 <= row["resolved_fraction"] <= 1
        assert 0 <= row["certified_top10_recall"] <= 1
        assert row["iterations_executed"] <= row["iteration_budget"]


def test_deliberately_invalid_bound_cannot_pass_the_independent_truth_check(monkeypatch):
    monkeypatch.setattr(experiment, "poisson_error_bound", lambda defect, a_min: np.zeros_like(defect))
    with pytest.raises(AssertionError, match="signed_bound_violations"):
        experiment.evaluate_field(*_field(), budgets=[1])


def test_bootstrap_uses_matched_field_rows_for_paired_localization():
    records = []
    for sample in range(2):
        records.extend({"sample": sample, **row} for row in experiment.evaluate_field(*_field(sample), budgets=[2]))
    frame = pd.DataFrame(records)
    result = experiment.summarize(frame, n_fields=2, bootstrap=100, seed=91)
    expected = (frame[frame.method == "poisson_pcg2"].spearman.to_numpy() -
                frame[frame.method == "raw_residual"].spearman.to_numpy()).mean()
    assert result["paired_spearman_differences"]["poisson_pcg2_minus_raw_residual"]["mean"] == pytest.approx(expected)
    assert "resolved_fraction" not in result["methods"]["raw_residual"]


def test_duplicate_requested_budgets_and_invalid_coefficients_are_rejected():
    coefficient, forcing, prediction, truth = _field()
    with pytest.raises(ValueError, match="unique"):
        experiment.evaluate_field(coefficient, forcing, prediction, truth, budgets=[2, 2])
    coefficient[0, 0] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        experiment.evaluate_field(coefficient, forcing, prediction, truth)
