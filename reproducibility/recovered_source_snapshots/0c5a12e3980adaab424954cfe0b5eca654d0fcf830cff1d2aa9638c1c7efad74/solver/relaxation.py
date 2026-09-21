"""Relaxation sweeps for the compute ladder partial inverse rungs.

The routines approximate A e = r using the same assembled sparse operator used by
the solver and residual code. They intentionally implement only the small sweep
counts used in the paper ladder so accidental expensive runs fail early.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

SUPPORTED_SWEEPS = {1, 2, 5, 10, 20, 40, 80}


def _validate_inputs(
    A: sp.spmatrix, r: np.ndarray, e0: np.ndarray, n_sweeps: int
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, tuple[int, ...]]:
    if isinstance(n_sweeps, bool) or not isinstance(n_sweeps, (int, np.integer)) or n_sweeps not in SUPPORTED_SWEEPS:
        raise ValueError(f"n_sweeps must be one of {sorted(SUPPORTED_SWEEPS)}; got {n_sweeps}")
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square; got {A.shape}")

    A_csr = A.tocsr()
    n = A_csr.shape[0]
    rhs = np.asarray(r, dtype=np.float64).reshape(-1)
    e_arr = np.asarray(e0, dtype=np.float64)
    e = e_arr.reshape(-1).copy()
    if rhs.size != n or e.size != n:
        raise ValueError(f"shape mismatch: A {A.shape}, r {rhs.shape}, e0 {e.shape}")
    if n == 0 or not all(np.all(np.isfinite(x)) for x in (rhs, e, A_csr.data)):
        raise ValueError("A, r, and e0 must be nonempty and finite")

    diag = A_csr.diagonal()
    if np.any(diag == 0.0) or not np.all(np.isfinite(diag)):
        raise ValueError("A has a zero or non-finite diagonal")
    return A_csr, rhs, e, e_arr.shape


def jacobi_sweep(A: sp.spmatrix, r: np.ndarray, e0: np.ndarray, n_sweeps: int) -> np.ndarray:
    """Run damp-free Jacobi sweeps for A e = r and return the final iterate."""
    A_csr, rhs, e, out_shape = _validate_inputs(A, r, e0, n_sweeps)
    d_inv = 1.0 / A_csr.diagonal()
    for _ in range(n_sweeps):
        e = e + d_inv * (rhs - A_csr.dot(e))
    return e.reshape(out_shape)


def gauss_seidel_sweep(A: sp.spmatrix, r: np.ndarray, e0: np.ndarray, n_sweeps: int) -> np.ndarray:
    """Run forward Gauss-Seidel sweeps for A e = r and return the final iterate.

    Unlike Jacobi, a forward sweep immediately uses updated values: it can
    propagate information through an entire directed grid path in one sweep.
    It is therefore not a fixed-radius local estimator.
    """
    A_csr, rhs, e, out_shape = _validate_inputs(A, r, e0, n_sweeps)
    indptr = A_csr.indptr
    indices = A_csr.indices
    data = A_csr.data

    for _ in range(n_sweeps):
        for row in range(A_csr.shape[0]):
            start, end = indptr[row], indptr[row + 1]
            diag = 0.0
            offdiag_dot = 0.0
            for ptr in range(start, end):
                col = indices[ptr]
                val = data[ptr]
                if col == row:
                    diag = val
                else:
                    offdiag_dot += val * e[col]
            if diag == 0.0:
                raise ValueError(f"A has a zero diagonal at row {row}")
            e[row] = (rhs[row] - offdiag_dot) / diag
    return e.reshape(out_shape)


def cg_correction(
    A: sp.spmatrix, r: np.ndarray, maxiter: int, *, diagonal_preconditioner: bool = False
) -> tuple[np.ndarray, dict[str, int | float | bool]]:
    """CG/diagonal-PCG correction from zero, with an *at most* iteration budget.

    A round-off tolerance allows early exact/near-exact convergence instead of
    asking SciPy to continue at zero tolerance and divide by zero. The returned
    metadata records iterations actually performed for honest cost accounting.
    ``info > 0`` means the budget expired; it is expected for this partial solve.

    Unpreconditioned CG has ``x_k in span(r, Ar, ..., A**(k-1)r)`` and expands
    the support of a sparse residual at most k-1 hops. Its global inner products
    mean it does NOT have bounded-radius dependence on residual values. The
    same distinction holds for diagonal PCG; Jacobi has local dependence.
    """
    if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    rhs_arr = np.asarray(r, dtype=np.float64)
    rhs = rhs_arr.reshape(-1)
    a = A.tocsr().astype(np.float64)
    if a.shape != (rhs.size, rhs.size) or rhs.size == 0:
        raise ValueError(f"shape mismatch: A {a.shape}, r {rhs_arr.shape}")
    if not np.all(np.isfinite(rhs)) or not np.all(np.isfinite(a.data)):
        raise ValueError("A and r must be finite")
    preconditioner = None
    if diagonal_preconditioner:
        diagonal = a.diagonal()
        if np.any(diagonal <= 0.0):
            raise ValueError("diagonal PCG requires a strictly positive diagonal")
        preconditioner = spla.LinearOperator(a.shape, matvec=lambda v: v / diagonal, dtype=np.float64)
    performed = 0

    def count_iteration(_x: np.ndarray) -> None:
        nonlocal performed
        performed += 1

    x, info = spla.cg(
        a, rhs, x0=np.zeros_like(rhs), rtol=8.0 * np.finfo(np.float64).eps,
        atol=0.0, maxiter=int(maxiter), M=preconditioner, callback=count_iteration,
    )
    if info < 0 or not np.all(np.isfinite(x)):
        raise FloatingPointError(f"CG failed with info={info}")
    return np.asarray(x, dtype=np.float64).reshape(rhs_arr.shape), {
        "iterations_requested": int(maxiter),
        "iterations_performed": performed,
        "solver_info": int(info),
        "diagonal_preconditioner": bool(diagonal_preconditioner),
    }
