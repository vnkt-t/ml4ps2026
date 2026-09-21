"""Proposition 1 counterexample: a small SPD M-matrix where the residual mislocalizes the error.

A is the stiffness matrix of a 1D variable-coefficient elliptic operator (TPFA, Dirichlet),
transmissibilities (t_bL, t_01, t_12, t_bR) = (2,1,1,3):
    A = [[3,-1,0],[-1,2,-1],[0,-1,4]]   (SPD M-matrix)
For e = (2,3,2)^T, r = A e = (3,2,5)^T. The largest-error cell (2) has the SMALLEST residual,
and the residual's largest entry is at cell 3 (smaller error) -> argmax|e| != argmax|r|, so |r|
is not a valid pointwise localizer of |e|; recovering it needs the (nonlocal) inverse e = A^{-1} r.

Found by brute-force search over small integer SPD M-matrices (smallest clean strict example).

Usage: .venv/bin/python analysis/counterexample.py [out_png]
"""
from __future__ import annotations
import os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(REPO, ".pytest_cache", "mpl"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

A = np.array([[3.0, -1.0, 0.0],
              [-1.0, 2.0, -1.0],
              [0.0, -1.0, 4.0]])
e = np.array([2.0, 3.0, 2.0])


def report():
    r = A @ e
    assert np.allclose(A, A.T), "A not symmetric"
    eig = np.linalg.eigvalsh(A)
    assert np.all(eig > 0), "A not SPD"
    offdiag = A - np.diag(np.diag(A))
    assert np.all(offdiag <= 0) and np.all(np.diag(A) > 0), "A not an M-matrix"
    recovered = np.linalg.solve(A, r)
    print("A =\n", A.astype(int))
    print("eigvalues(A) =", np.round(eig, 4), "(SPD)")
    print("e        =", e.astype(int), "  argmax|e| = cell", int(np.argmax(np.abs(e))) + 1)
    print("r = A e  =", r.astype(int), "  argmax|r| = cell", int(np.argmax(np.abs(r))) + 1,
          "  argmin|r| = cell", int(np.argmin(np.abs(r))) + 1)
    print("A^{-1} r =", np.round(recovered, 6), "(== e, inverse recovers the error)")
    assert np.allclose(recovered, e)
    assert np.argmax(np.abs(e)) != np.argmax(np.abs(r)), "no mislocalization"
    assert np.argmin(np.abs(r)) == np.argmax(np.abs(e)), "max-error cell is not min-residual"
    print("CHECK PASS: argmax|e|=cell2 (largest error) is argmin|r| (smallest residual);"
          " argmax|r|=cell3.")
    return r


def figure(out_png: str):
    r = A @ e
    import matplotlib.pyplot as plt
    cells = np.arange(1, 4)
    w = 0.38
    fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    b1 = ax.bar(cells - w / 2, np.abs(e), w, label="|e|  (true error)", color="#4c72b0")
    b2 = ax.bar(cells + w / 2, np.abs(r), w, label="|r| = |Ae|  (residual)", color="#c44e52")
    ax.set_xticks(cells); ax.set_xticklabels([f"cell {i}" for i in cells])
    ax.set_ylabel("magnitude"); ax.set_ylim(0, 6.4)
    ax.set_title("Residual mislocalizes the error\n"
                 r"$\arg\max_i|e_i|=$cell 2 $\neq$ cell 3$=\arg\max_i|r_i|$", fontsize=9.5)
    ax.legend(fontsize=8, loc="upper left")
    # annotate the punchline cells
    ax.annotate("largest error,\nsmallest residual", (2 - w / 2, abs(e[1])),
                textcoords="offset points", xytext=(-6, 6), fontsize=7.2, color="#4c72b0")
    ax.annotate("residual peak,\nsmaller error", (3 + w / 2, abs(r[2])),
                textcoords="offset points", xytext=(-58, -22), fontsize=7.2, color="#c44e52")
    fig.savefig(out_png, dpi=150)
    print(f"saved {out_png}")


if __name__ == "__main__":
    report()
    out = sys.argv[1] if len(sys.argv) > 1 else "paper/figures/fig0_counterexample.png"
    figure(os.path.join(REPO, out) if not os.path.isabs(out) else out)
