"""T0.5 pilot: train a tiny FNO ENSEMBLE on solved Darcy data (fixed smooth source f).

Saves test predictions + ensemble std for the T0.6 nested-predictor pilot. f is constant
(independent of a), so the FNO error field is realistic and not confounded by the source.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generate_grf import make_dataset  # noqa: E402
from data.generate_contrast import family_C_sample  # noqa: E402
from models.tiny_fno import FNO2d  # noqa: E402


def build_inputs(a, logmean, logstd, N):
    la = (np.log(a) - logmean) / logstd
    xs = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    n = a.shape[0]
    Xb = np.broadcast_to(X, (n, N, N))
    Yb = np.broadcast_to(Y, (n, N, N))
    return np.stack([la, Xb, Yb], axis=-1).astype(np.float32)


def rel_l2(pred, true):
    n = pred.shape[0]
    num = np.linalg.norm((pred - true).reshape(n, -1), axis=1)
    den = np.linalg.norm(true.reshape(n, -1), axis=1)
    return num / den


def make_contrast_dataset(N, n, kappa, rng):
    a = np.empty((n, N, N), dtype=np.float64)
    f = np.empty((n, N, N), dtype=np.float64)
    u = np.empty((n, N, N), dtype=np.float64)
    for i in range(n):
        sample = family_C_sample(N=N, kappa=kappa, rng=rng)
        a[i] = sample.a
        f[i] = sample.f
        u[i] = sample.u_true
        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f"generated contrast kappa={kappa:g}: {i + 1}/{n}", flush=True)
    return a, f, u


def train_one(seed, Xtr, Ytr, Xte, args, device):
    torch.manual_seed(seed)
    model = FNO2d(modes=args.modes, width=args.width, n_layers=args.layers, in_ch=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    xtr = torch.tensor(Xtr, device=device)
    ytr = torch.tensor(Ytr, device=device)
    n, bs = xtr.shape[0], 32
    lossfn = torch.nn.MSELoss()
    for _ in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossfn(model(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(Xte, device=device)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--ntrain", type=int, default=700)
    ap.add_argument("--ntest", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--family", choices=["grf", "contrast"], default=None)
    ap.add_argument("--kappa", type=float, default=None)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results/pilot_fno.npz")
    args = ap.parse_args()
    if args.smoke:
        args.ntrain, args.ntest, args.seeds, args.epochs = 40, 20, 1, 3
    family = args.family or ("contrast" if args.kappa is not None else "grf")
    if family == "contrast" and args.kappa is None:
        args.kappa = 100.0
    if family == "contrast" and args.out == "results/pilot_fno.npz":
        kappa_label = f"{args.kappa:g}".replace(".", "p")
        args.out = f"results/fno_ensemble_kappa{kappa_label}.npz"

    # FFT-based FNO: MPS rfft2 support is unreliable on torch; 32x32 is fast on CPU.
    device = "cpu"
    t0 = time.time()
    rng = np.random.default_rng(args.data_seed)
    f_tr = None
    f_te = None
    if family == "contrast":
        a_tr, f_tr, u_tr = make_contrast_dataset(args.N, args.ntrain, float(args.kappa), rng)
        a_te, f_te, u_te = make_contrast_dataset(args.N, args.ntest, float(args.kappa), rng)
    else:
        a_tr, u_tr = make_dataset(args.N, args.ntrain, rng)
        a_te, u_te = make_dataset(args.N, args.ntest, rng)

    logmean, logstd = float(np.log(a_tr).mean()), float(np.log(a_tr).std())
    umean, ustd = float(u_tr.mean()), float(u_tr.std())
    Xtr = build_inputs(a_tr, logmean, logstd, args.N)
    Xte = build_inputs(a_te, logmean, logstd, args.N)
    Ytr = ((u_tr - umean) / ustd).astype(np.float32)

    preds = np.empty((args.seeds, args.ntest, args.N, args.N), dtype=np.float32)
    for s in range(args.seeds):
        p_phys = train_one(s, Xtr, Ytr, Xte, args, device) * ustd + umean
        preds[s] = p_phys
        r = rel_l2(p_phys, u_te)
        print(f"seed {s}: test rel-L2 mean {r.mean():.4f} median {np.median(r):.4f}", flush=True)

    ens = preds.mean(0)
    er = rel_l2(ens, u_te)
    sigma = preds.std(0)
    print(f"ENSEMBLE rel-L2 mean {er.mean():.4f} median {np.median(er):.4f}", flush=True)
    print(f"sigma_ens mean {sigma.mean():.3e} | {time.time()-t0:.1f}s | device {device}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {
        "a_test": a_te,
        "u_test": u_te,
        "u_pred": preds,
        "sigma_ens": sigma,
        "family": family,
        "logmean": logmean,
        "logstd": logstd,
        "umean": umean,
        "ustd": ustd,
    }
    if family == "contrast":
        payload["kappa"] = float(args.kappa)
        payload["f_test"] = f_te
    else:
        payload["f_source"] = 1.0
    np.savez_compressed(args.out, **payload)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
