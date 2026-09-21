"""Regression tests for partial-solve support, termination, and actual cost."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from solver.assemble_fd import assemble
from solver import multigrid
from solver.relaxation import cg_correction, gauss_seidel_sweep, jacobi_sweep
from uncertainty.compute_ladder import compute_ladder


def test_cg_remote_values_change_local_iterate_despite_disconnected_support():
    A = sp.diags([2.0, 100.0], format="csr")
    local, _ = cg_correction(A, np.array([1.0, 0.0]), 1)
    remote, _ = cg_correction(A, np.array([1.0, 1.0]), 1)
    assert local[0] == pytest.approx(0.5)
    assert remote[0] == pytest.approx(2.0 / 102.0)
    # Jacobi truly has local information dependence.
    jlocal = jacobi_sweep(A, [1.0, 0.0], np.zeros(2), 1)
    jremote = jacobi_sweep(A, [1.0, 1.0], np.zeros(2), 1)
    assert jlocal[0] == jremote[0]


def test_sparse_residual_support_and_gauss_seidel_directionality():
    n = 17
    A = sp.diags([-np.ones(n-1), 2*np.ones(n), -np.ones(n-1)], [-1, 0, 1], format="csr")
    rhs = np.zeros(n)
    rhs[0] = 1.0
    cg, _ = cg_correction(A, rhs, 5)
    jac = jacobi_sweep(A, rhs, np.zeros(n), 5)
    gs = gauss_seidel_sweep(A, rhs, np.zeros(n), 1)
    assert np.count_nonzero(cg[5:]) == 0
    assert np.count_nonzero(jac[5:]) == 0
    assert np.all(gs > 0.0)  # one forward sweep reaches the entire path


@pytest.mark.parametrize("precondition", [False, True])
def test_cg_early_exact_convergence_is_finite_and_costed(precondition):
    A = sp.eye(9, format="csr")
    rhs = np.arange(9.0)
    x, info = cg_correction(A, rhs, 20, diagonal_preconditioner=precondition)
    np.testing.assert_array_equal(x, rhs)
    assert info["iterations_performed"] == 1
    zero, info = cg_correction(A, np.zeros(9), 20, diagonal_preconditioner=precondition)
    np.testing.assert_array_equal(zero, np.zeros(9))
    assert info["iterations_performed"] == 0


def test_amg_first_cycle_is_charged_and_warm_benchmark_separate(monkeypatch):
    pyamg = pytest.importorskip("pyamg")
    now = [0.0]

    class FakeHierarchy:
        levels = [SimpleNamespace(A=sp.eye(4)), SimpleNamespace(A=sp.eye(1))]
        calls = 0

        def solve(self, rhs, **_kwargs):
            now[0] += 5.0 if self.calls == 0 else 2.0
            self.calls += 1
            return rhs.copy()

        def cycle_complexity(self):
            return 2.25

        def operator_complexity(self):
            return 1.25

    def setup(*_args, **_kwargs):
        now[0] += 3.0
        return FakeHierarchy()

    monkeypatch.setattr(pyamg, "smoothed_aggregation_solver", setup)
    monkeypatch.setattr(multigrid.time, "perf_counter", lambda: now[0])
    monkeypatch.setattr(multigrid, "_time_matvec", lambda *_args: 0.5)
    _, cost = multigrid._pyamg_vcycle(sp.eye(4, format="csr"), np.ones(4), None)
    assert cost["setup_s"] == 3.0
    assert cost["first_cycle_s"] == 5.0
    assert cost["cycle_s"] == 2.0
    assert cost["total_s"] == 8.0
    assert cost["total_matvec_equiv"] == 16.0
    assert cost["schema_version"] == 2


def test_failed_amg_cost_is_retained_before_fallback(monkeypatch):
    monkeypatch.setattr(multigrid, "pyamg_available", lambda: True)
    monkeypatch.setattr(multigrid, "_pyamg_vcycle", lambda A, r, levels: (
        np.zeros_like(r), {"total_s": 10.0, "matvec_s": 0.1, "cycle_complexity": 3.0}
    ))
    monkeypatch.setattr(multigrid, "_time_matvec", lambda *_args: 0.1)
    x = multigrid.vcycle(sp.eye(4, format="csr"), np.ones(4))
    np.testing.assert_allclose(x, 1.0)
    cost = multigrid.last_vcycle_cost()
    assert cost["failed_amg_s"] >= 10.0
    assert cost["total_s"] >= cost["failed_amg_s"]
    assert cost["total_matvec_equiv"] >= 100.0
    assert cost["cycle_complexity"] == 43.0
    assert multigrid.last_vcycle_backend() == "pyamg_fallback_gs_40"


def test_zero_rhs_does_not_accept_nonzero_amg_correction():
    A = sp.eye(4, format="csr")
    assert multigrid._residual_reduced(A, np.zeros(4), np.zeros(4))
    assert not multigrid._residual_reduced(A, np.ones(4), np.zeros(4))


def test_high_contrast_ladder_is_finite_and_oracle_recovers_error():
    rng = np.random.default_rng(13)
    n = 10
    a = np.ones((n, n))
    a[2:7, 3:8] = 500.0
    A = assemble(a)
    error = rng.normal(size=(n, n))
    ladder = compute_ladder(A, error, 0.0, np.zeros_like(error), None, None)
    for rung in ladder.values():
        assert np.all(np.isfinite(rung["score"]))
        assert rung["time_s"] >= 0.0
    np.testing.assert_allclose(ladder["5_oracle"]["score"], np.abs(error), rtol=2e-10, atol=1e-12)
    cost = ladder["4_vcycle"]["amg_cost"]
    assert cost["schema_version"] == 2
    if ladder["4_vcycle"]["backend"] == "pyamg_smoothed_aggregation":
        assert cost["total_s"] >= cost["setup_s"] + cost["first_cycle_s"]
    else:
        assert cost["total_s"] >= cost["fallback_s"]
