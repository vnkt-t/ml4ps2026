"""Adaptive stopping decisions use only enclosures, never an error oracle."""
import numpy as np
import pytest
import scipy.sparse.linalg as spla

from solver.adaptive import adaptive_poisson_cg
from solver.assemble_fd import assemble
from solver.poisson import poisson_preconditioner
from solver import poisson as poisson_module


@pytest.mark.parametrize("bound_kind", ["poisson", "flux"])
def test_adaptive_constant_coefficient_stops_at_convergence_without_oracle(monkeypatch, bound_kind):
    n = 10
    rng = np.random.default_rng(66)
    truth = rng.normal(size=(n, n))
    A = assemble(np.ones((n, n)))
    rhs = A @ truth.reshape(-1)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("adaptive correction must not call a direct-solve oracle")

    monkeypatch.setattr(spla, "spsolve", forbidden)
    monkeypatch.setattr(spla, "splu", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(np.linalg, "inv", forbidden)
    result = adaptive_poisson_cg(A, rhs, 1.0, bound_kind=bound_kind, coefficient=np.ones((n, n)))
    assert result.target_met and result.status == "rank_target"
    assert result.iterations == 1  # final check occurs before the five-step checkpoint
    assert result.certified_high_recall >= 0.9
    assert result.target_high_count == 9
    assert result.cg_operator_applications == result.preconditioner_applications == 1
    assert result.extra_operator_applications == result.bound_poisson_solves == 1
    assert result.fft_transforms == 4
    assert result.bound_kind == bound_kind
    np.testing.assert_allclose(result.correction, truth.reshape(-1), atol=1e-13)


def test_adaptive_target_count_and_rank_flags_are_correct_on_heterogeneous_field():
    n = 12
    rng = np.random.default_rng(67)
    a = np.where(rng.random((n, n)) < 0.35, 100.0, 1.0)
    truth = rng.normal(size=(n, n))
    A = assemble(a)
    result = adaptive_poisson_cg(A, (A @ truth.reshape(-1)).reshape(n, n), 1.0)
    assert result.target_met
    assert result.target_high_count == int(np.ceil(0.9 * np.ceil(0.1 * n*n)))
    assert result.high.sum() >= result.target_high_count
    assert result.iterations <= 80
    assert all(item["certified_high_count"] < result.target_high_count for item in result.history[:-1])
    top = np.zeros(n*n, dtype=bool)
    top[np.argsort(np.abs(truth).reshape(-1))[-result.n_top:]] = True
    assert not np.any(result.high.reshape(-1) & ~top)
    assert not np.any(result.low.reshape(-1) & top)
    assert result.extra_operator_applications == result.bound_poisson_solves == len(result.history)
    assert result.cg_operator_applications == result.preconditioner_applications == result.iterations
    fixed, _ = spla.cg(A, A @ truth.reshape(-1), M=poisson_preconditioner(n),
                       x0=np.zeros(n*n), maxiter=result.iterations, atol=0.0,
                       rtol=8.0 * np.finfo(float).eps)
    np.testing.assert_allclose(result.correction.reshape(-1), fixed, rtol=1e-12, atol=1e-13)


def test_maximum_budget_reports_unmet_target():
    n = 10
    rng = np.random.default_rng(68)
    a = np.where(rng.random((n, n)) < 0.4, 500.0, 1.0)
    A = assemble(a)
    rhs = rng.normal(size=(n, n))
    result = adaptive_poisson_cg(A, rhs, 1.0, maxiter=1, target_recall=1.0)
    assert result.iterations == 1
    assert not result.target_met and result.status == "iteration_limit"
    assert result.bound_poisson_solves == 1  # final check even without a scheduled checkpoint


def test_zero_residual_ties_remain_unresolved_without_wasted_iterations():
    A = assemble(np.ones((6, 6)))
    result = adaptive_poisson_cg(A, np.zeros((6, 6)), 1.0)
    assert result.iterations == result.cg_operator_applications == result.preconditioner_applications == 0
    assert result.status == "converged_unresolved" and not result.target_met
    assert not result.high.any() and not result.low.any()
    np.testing.assert_array_equal(result.bound, 0.0)
    assert result.bound_poisson_solves == result.extra_operator_applications == 1


def test_top_all_is_trivial_even_with_ties():
    A = assemble(np.ones((3, 3)))
    result = adaptive_poisson_cg(A, np.zeros((3, 3)), 1.0, q=1.0)
    assert result.target_met and result.high.all() and not result.low.any()
    assert result.iterations == 0


def test_flux_stopping_is_no_later_and_accounts_for_the_same_check_cost():
    n = 12
    rng = np.random.default_rng(69)
    a = np.where(rng.random((n, n)) < 0.4, 500.0, 1.0)
    A = assemble(a)
    truth = rng.normal(size=(n, n))
    rhs = A @ truth.reshape(-1)
    baseline = adaptive_poisson_cg(A, rhs, 1.0)
    flux = adaptive_poisson_cg(A, rhs, 1.0, bound_kind="flux", coefficient=a)
    assert baseline.target_met and flux.target_met
    assert flux.iterations <= baseline.iterations
    assert flux.bound_poisson_solves == flux.extra_operator_applications == len(flux.history)
    assert flux.fft_transforms == 2 * (flux.iterations + len(flux.history))
    top = np.zeros(n*n, dtype=bool)
    top[np.argsort(np.abs(truth).reshape(-1))[-flux.n_top:]] = True
    assert not np.any(flux.high.reshape(-1) & ~top)
    assert not np.any(flux.low.reshape(-1) & top)


def test_flux_requires_observed_coefficient_and_valid_lower_bound():
    A = assemble(np.ones((2, 2)))
    with pytest.raises(ValueError, match="requires"):
        adaptive_poisson_cg(A, np.ones(4), 1.0, bound_kind="flux")
    with pytest.raises(ValueError, match="exceeds"):
        adaptive_poisson_cg(A, np.ones(4), 2.0, bound_kind="flux", coefficient=np.ones((2, 2)))


@pytest.mark.parametrize("bound_kind", ["poisson", "flux"])
def test_reported_fft_work_matches_actual_transform_calls(monkeypatch, bound_kind):
    calls = {"forward": 0, "inverse": 0}
    forward, inverse = poisson_module.dstn, poisson_module.idstn

    def counted_forward(*args, **kwargs):
        calls["forward"] += 1
        return forward(*args, **kwargs)

    def counted_inverse(*args, **kwargs):
        calls["inverse"] += 1
        return inverse(*args, **kwargs)

    monkeypatch.setattr(poisson_module, "dstn", counted_forward)
    monkeypatch.setattr(poisson_module, "idstn", counted_inverse)
    rng = np.random.default_rng(70)
    a = np.where(rng.random((8, 8)) < 0.4, 100.0, 1.0)
    result = adaptive_poisson_cg(assemble(a), rng.normal(size=(8, 8)), 1.0,
                                 bound_kind=bound_kind, coefficient=a)
    assert result.fft_transforms == calls["forward"] + calls["inverse"]
    assert calls["forward"] == result.preconditioner_applications + result.bound_poisson_solves


@pytest.mark.parametrize("options", [
    {"maxiter": 0}, {"check_every": True}, {"target_recall": 0.0}, {"q": 1.1},
])
def test_invalid_adaptive_protocol_fails(options):
    with pytest.raises(ValueError):
        adaptive_poisson_cg(assemble(np.ones((2, 2))), np.ones(4), 1.0, **options)
