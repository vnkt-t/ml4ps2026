"""Is the sigma_ens-vs-raw-|r| ordering an artifact of one lucky 3-seed ensemble?

The headline FNO result uses a 3-member ensemble, and the bootstrap CIs resample TEST FIELDS,
not trained models. They therefore quantify variation over test PDEs conditional on those three
networks; they say nothing about training randomness. The claim most exposed to that gap is the
ranking claim "sigma_ens outranks raw |r| at kappa >= 100", because sigma_ens is itself a
function of which models are in the ensemble.

This trains nothing. It loads 5 saved checkpoints per kappa and recomputes the two cheap scores
for the full 5-member ensemble and for all C(5,3)=10 three-member subsets, so the spread over
ensemble composition is visible directly. Rungs requiring AMG or relaxation are deliberately
excluded: they are expensive and are not the claim at risk.

Usage: .venv/bin/python analysis/check_seed_stability.py
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from experiments.exp0_pilot_fno import make_contrast_dataset  # noqa: E402
from experiments.exp_real_fno_multi_kappa import load_ensemble  # noqa: E402
from solver.assemble_fd import assemble  # noqa: E402

KAPPAS = [10.0, 100.0, 500.0]
SEEDS = [0, 1, 2, 3, 4]
SUBSET_SIZE = 3
N_TRAIN, N_TEST, GRID_N, DATA_SEED = 700, 200, 32, 0
CKPT_DIR = Path(REPO) / "results" / "checkpoints_5seed"


def scores_for(members: np.ndarray, idx: tuple[int, ...], A_list, u_true, f_rhs):
    """Return (spearman(sigma_ens,|e|), spearman(|r|,|e|)) for the ensemble subset `idx`."""
    sub = members[list(idx)]
    u_hat = sub.mean(axis=0)
    sigma = sub.std(axis=0)
    sig_rho, raw_rho = [], []
    for i, A in enumerate(A_list):
        e = np.abs((u_hat[i] - u_true[i]).reshape(-1))
        r = np.abs(A @ u_hat[i].reshape(-1) - f_rhs[i].reshape(-1))
        sig_rho.append(spearmanr(sigma[i].reshape(-1), e).statistic)
        raw_rho.append(spearmanr(r, e).statistic)
    return float(np.mean(sig_rho)), float(np.mean(raw_rho))


def main() -> None:
    out: dict[str, dict] = {}
    for kappa in KAPPAS:
        rng = np.random.default_rng(DATA_SEED)
        make_contrast_dataset(GRID_N, N_TRAIN, kappa, rng)  # advance rng exactly as training did
        a_te, f_te, u_te = make_contrast_dataset(GRID_N, N_TEST, kappa, rng)

        ens = load_ensemble(
            kappa, seeds=SEEDS, checkpoint_dir=CKPT_DIR, grid_N=GRID_N,
            n_train=N_TRAIN, n_test=N_TEST, data_seed=DATA_SEED,
        )
        members = np.stack([np.stack([m(a_te[i]) for i in range(N_TEST)]) for m in ens])
        A_list = [assemble(a_te[i]).tocsr() for i in range(N_TEST)]

        subsets = list(itertools.combinations(range(len(SEEDS)), SUBSET_SIZE))
        rows = []
        for idx in subsets:
            s, r = scores_for(members, idx, A_list, u_te, f_te)
            rows.append({"seeds": [SEEDS[j] for j in idx], "sigma_rho": s, "raw_r_rho": r,
                         "sigma_beats_raw": bool(s > r)})
        full_s, full_r = scores_for(members, tuple(range(len(SEEDS))), A_list, u_te, f_te)

        n_beat = sum(r["sigma_beats_raw"] for r in rows)
        out[f"{kappa}"] = {
            "kappa": kappa,
            "n_subsets": len(rows),
            "subset_size": SUBSET_SIZE,
            "subsets": rows,
            "subset_sigma_rho_min": min(r["sigma_rho"] for r in rows),
            "subset_sigma_rho_max": max(r["sigma_rho"] for r in rows),
            "subset_raw_rho_min": min(r["raw_r_rho"] for r in rows),
            "subset_raw_rho_max": max(r["raw_r_rho"] for r in rows),
            "subsets_where_sigma_beats_raw": n_beat,
            "full5_sigma_rho": full_s,
            "full5_raw_r_rho": full_r,
            "full5_sigma_beats_raw": bool(full_s > full_r),
        }
        print(f"kappa={kappa:g}  ({len(rows)} three-seed subsets of {len(SEEDS)})")
        print(f"  sigma_ens rho  subsets [{out[f'{kappa}']['subset_sigma_rho_min']:.3f},"
              f" {out[f'{kappa}']['subset_sigma_rho_max']:.3f}]   full-5 {full_s:.3f}")
        print(f"  raw |r|   rho  subsets [{out[f'{kappa}']['subset_raw_rho_min']:.3f},"
              f" {out[f'{kappa}']['subset_raw_rho_max']:.3f}]   full-5 {full_r:.3f}")
        print(f"  sigma_ens > raw |r| in {n_beat}/{len(rows)} subsets; full-5: "
              f"{out[f'{kappa}']['full5_sigma_beats_raw']}", flush=True)

    path = os.path.join(REPO, "results", "check_seed_stability.json")
    with open(path, "w") as fh:
        json.dump({"grid_N": GRID_N, "seeds": SEEDS, "n_test": N_TEST, "per_kappa": out}, fh, indent=2)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
