"""Second-architecture robustness check: does the residual-localization gap depend on the FNO?

Trains a 3-seed tiny U-Net (purely local convolutions) on the IDENTICAL Family-C kappa=100, 32x32
data the FNO saw (same data_seed/n_train/n_test -> byte-identical test fields), then runs the SAME
compute ladder and the SAME localization metrics with bootstrap CIs. Apples-to-apples; only the
architecture changes. The locked FNO pipeline is untouched.

Usage: .venv/bin/python experiments/exp_arch_check_unet.py [--epochs 80] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(REPO, ".pytest_cache", "mpl"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from data.generate_contrast import family_C_sample  # noqa: E402
from models.tiny_unet import UNet2d  # noqa: E402
from solver.assemble_fd import assemble  # noqa: E402
from uncertainty.compute_ladder import compute_ladder  # noqa: E402
from experiments.exp0_pilot_fno import build_inputs, make_contrast_dataset, rel_l2  # noqa: E402
from experiments.exp_real_fno_multi_kappa import (  # noqa: E402
    bootstrap_mean_ci, localization_metrics_top10, predict_model_physical,
)

# The headline rungs, matching the FNO tables.
RUNGS = {
    "0_sigma_ens": "sigma_ens",
    "1_abs_r": "raw |r|",
    "3e_jacobi_20": "Jacobi-20",
    "4_vcycle": "AMG V-cycle",
    "5_oracle": "oracle",
}
# FNO kappa=100 reference (results/real_fno_multi_kappa_amgfix.json) for the side-by-side print.
# rung 4 updated 2026-08-21 with the full AMG hierarchy (was 0.978 under the old shallow
# two-level cycle, which solved 17% of the unknowns exactly and mispriced the rung).
FNO_REF = {"0_sigma_ens": 0.316, "1_abs_r": 0.248, "3e_jacobi_20": 0.514,
           "4_vcycle": 0.911, "5_oracle": 1.000}


def train_unet(seed, Xtr, Ytr, args, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = UNet2d(width=args.width, in_ch=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    xtr = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr = torch.tensor(Ytr, dtype=torch.float32, device=device)
    n, bs = xtr.shape[0], args.batch_size
    lossfn = torch.nn.MSELoss()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossfn(model(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        sched.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--kappa", type=float, default=100.0)
    ap.add_argument("--ntrain", type=int, default=700)
    ap.add_argument("--ntest", type=int, default=200)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.ntrain, args.ntest, args.seeds = 2, 40, 20, [0]
    device = "cpu"
    t0 = time.time()

    # IDENTICAL data to the FNO kappa=100 run: same seed, same draw order (train then test).
    rng = np.random.default_rng(int(args.data_seed))
    print(f"[arch-check U-Net] generating Family-C kappa={args.kappa:g} (same fields as FNO)", flush=True)
    a_tr, _f_tr, u_tr = make_contrast_dataset(args.N, args.ntrain, float(args.kappa), rng)
    a_te, f_te, u_te = make_contrast_dataset(args.N, args.ntest, float(args.kappa), rng)
    fp = __import__("hashlib").sha1(np.ascontiguousarray(a_te, dtype=np.float64).tobytes()).hexdigest()[:12]
    print(f"  test-field fingerprint={fp} (should match the FNO run's)", flush=True)

    norm = {"logmean": float(np.log(a_tr).mean()), "logstd": float(np.log(a_tr).std()),
            "umean": float(u_tr.mean()), "ustd": float(u_tr.std())}
    Xtr = build_inputs(a_tr, norm["logmean"], norm["logstd"], args.N)
    Xte = build_inputs(a_te, norm["logmean"], norm["logstd"], args.N)
    Ytr = ((u_tr - norm["umean"]) / norm["ustd"]).astype(np.float32)

    members = np.empty((len(args.seeds), args.ntest, args.N, args.N), dtype=np.float32)
    for j, s in enumerate(args.seeds):
        model = train_unet(int(s), Xtr, Ytr, args, device)
        members[j] = predict_model_physical(model, Xte, norm, device)
        rl = rel_l2(members[j].astype(np.float64), u_te.astype(np.float64))
        print(f"  U-Net seed={s}: rel-L2 mean={rl.mean():.4f}", flush=True)

    pred_mean = members.mean(axis=0)
    sigma = members.std(axis=0) if members.shape[0] > 1 else np.zeros_like(members[0])
    rl_ens = rel_l2(pred_mean.astype(np.float64), u_te.astype(np.float64))

    # Per-sample ladder + localization metrics (same A reused for everything).
    per = {k: {"spearman": [], "auroc": []} for k in RUNGS}
    backends = set()
    for i in range(args.ntest):
        A = assemble(a_te[i])
        u_hat = pred_mean[i].astype(np.float64)
        ladder = compute_ladder(A, u_hat, f_te[i], u_te[i],
                                members[:, i].astype(np.float64), sigma[i].astype(np.float64))
        abs_e = np.abs(u_hat - u_te[i])
        backends.add(ladder["4_vcycle"].get("backend", ""))
        for k in RUNGS:
            m = localization_metrics_top10(ladder[k]["score"], abs_e)
            per[k]["spearman"].append(m["spearman"])
            per[k]["auroc"].append(m["auroc"])
        if (i + 1) % 50 == 0:
            print(f"  ladder {i + 1}/{args.ntest}", flush=True)

    boot = np.random.default_rng(20260606)
    rows = {}
    for k in RUNGS:
        rows[k] = {
            "label": RUNGS[k],
            "spearman": bootstrap_mean_ci(np.array(per[k]["spearman"]), boot, args.n_boot),
            "auroc": bootstrap_mean_ci(np.array(per[k]["auroc"]), boot, args.n_boot),
        }

    out = {
        "experiment": "arch_check_unet",
        "architecture": "tiny_unet_2level_localconv",
        "kappa": args.kappa, "grid_N": args.N, "seeds": args.seeds,
        "data_seed": args.data_seed, "n_train": args.ntrain, "n_test": args.ntest,
        "test_field_fingerprint": fp,
        "rung4_backend": sorted(backends),
        "rel_l2_ens_mean": float(rl_ens.mean()), "rel_l2_ens_median": float(np.median(rl_ens)),
        "rungs": rows, "wall_time_s": time.time() - t0,
    }
    suffix = "_smoke" if args.smoke else ""
    out_json = os.path.join(REPO, f"results/arch_check_unet_kappa100{suffix}.json")
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n=== U-Net vs FNO (Spearman rho(score,|e|), kappa=100, 32x32) ===")
    print(f"{'rung':<14}{'U-Net':>10}{'  CI':>16}   {'FNO ref':>8}")
    for k in RUNGS:
        sp = rows[k]["spearman"]
        ci = f"[{sp['ci95_low']:.3f},{sp['ci95_high']:.3f}]"
        print(f"{RUNGS[k]:<14}{sp['mean']:>10.3f}{ci:>16}   {FNO_REF[k]:>8.3f}")
    print(f"\nU-Net ensemble rel-L2 mean={rl_ens.mean():.4f}  rung4_backend={sorted(backends)}")
    print(f"saved {out_json}  ({out['wall_time_s']:.0f}s)")


if __name__ == "__main__":
    main()
