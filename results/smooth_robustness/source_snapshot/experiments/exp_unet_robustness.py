"""Fixed-epoch U-Net robustness on the FNO's training and fresh evaluation fields.

No fresh-field labels enter fitting, normalization, checkpoint selection, or
training duration. This is a small architecture sensitivity check, not SOTA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import scipy
from scipy.sparse.linalg import cg, LinearOperator, spsolve
import torch

from analysis.metrics import localization_metrics_top10
from data.generate_contrast import family_C_sample
from experiments.exp0_pilot_fno import build_inputs
from models.tiny_unet import UNet2d
from solver.assemble_fd import assemble
from solver.multigrid import vcycle, last_vcycle_backend, last_vcycle_cost
from solver.poisson import poisson_preconditioner


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(path)


def fit_normalization(a_train, u_train):
    """Fit physical normalization from the two training arrays only."""
    a, u = np.asarray(a_train, dtype=np.float64), np.asarray(u_train, dtype=np.float64)
    if a.ndim != 3 or a.shape != u.shape or a.shape[1] != a.shape[2] or a.size == 0:
        raise ValueError("training arrays must be matching nonempty square fields")
    if not np.isfinite(a).all() or np.any(a <= 0) or not np.isfinite(u).all():
        raise ValueError("coefficients must be finite positive and targets finite")
    logs = np.log(a)
    norm = {"logmean": float(logs.mean()), "logstd": float(logs.std()),
            "umean": float(u.mean()), "ustd": float(u.std())}
    if norm["logstd"] <= 0 or norm["ustd"] <= 0:
        raise ValueError("training coefficient and target standard deviations must be positive")
    return norm


def encode_coefficients(a, normalization):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 3 or a.shape[1] != a.shape[2] or a.size == 0 or not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError("coefficients must be finite positive square fields")
    if not np.isfinite(list(normalization.values())).all() or normalization["logstd"] <= 0 or normalization["ustd"] <= 0:
        raise ValueError("normalization must be finite with positive standard deviations")
    return build_inputs(a, normalization["logmean"], normalization["logstd"], a.shape[1])


def predict_physical(model, a, normalization, batch_size=32):
    """Frozen evaluation, using saved training statistics for every batch."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    inputs = encode_coefficients(a, normalization)
    model.eval()
    blocks = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            output = model(torch.from_numpy(inputs[start:start+batch_size])).cpu().numpy()
            blocks.append(output.astype(np.float64)*normalization["ustd"]+normalization["umean"])
    result = np.concatenate(blocks)
    if result.shape != np.asarray(a).shape or not np.isfinite(result).all():
        raise FloatingPointError("invalid model prediction")
    return result


def train_member(seed, a_train, u_train, normalization, args, epochs_to_run=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = UNet2d(width=args.width, in_ch=3)
    x = torch.from_numpy(encode_coefficients(a_train, normalization))
    y = torch.from_numpy(((u_train-normalization["umean"])/normalization["ustd"]).astype(np.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []
    for epoch in range(args.epochs if epochs_to_run is None else epochs_to_run):
        start = time.perf_counter()
        model.train()
        order = torch.randperm(len(x))
        total = 0.0
        for offset in range(0, len(x), args.batch_size):
            indices = order[offset:offset+args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x[indices]), y[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite training loss, seed {seed}, epoch {epoch+1}")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        scheduler.step()
        history.append({"epoch": epoch+1, "normalized_training_mse": total/len(x),
                        "wall_time_s": time.perf_counter()-start})
        if epoch < 3 or (epoch+1) % 20 == 0 or epoch+1 == args.epochs:
            print(f"seed={seed} epoch={epoch+1}/{args.epochs} training_mse={history[-1]['normalized_training_mse']:.6f} seconds={history[-1]['wall_time_s']:.2f}", flush=True)
    return model, history


def generate_training(args):
    rng = np.random.default_rng(0)
    samples = [family_C_sample(N=32, kappa=100, rng=rng) for _ in range(args.n_train)]
    a, u = np.stack([s.a for s in samples]), np.stack([s.u_true for s in samples])
    return a, u


def mean_ci(values, draws):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("CI values must be finite scalars per field")
    lo, hi = np.quantile(values[draws].mean(axis=1), [.025, .975])
    return {"mean": float(values.mean()), "ci95_low": float(lo), "ci95_high": float(hi)}


def evaluate(a, f, truth, members, out, bootstrap):
    prediction, sigma = members.astype(np.float64).mean(axis=0), members.astype(np.float64).std(axis=0)
    rows, relative_errors, identities, costs = [], [], [], []
    M_poisson = poisson_preconditioner(32)
    for i in range(len(a)):
        A = assemble(a[i])
        rhs = A @ prediction[i].reshape(-1)-f[i].reshape(-1)
        diagonal = A.diagonal()
        M_diag = LinearOperator(A.shape, matvec=lambda x: x/diagonal, dtype=np.float64)
        corrections = {}
        executed = {}
        for name, M in [("cg20", None), ("diagonal_pcg20", M_diag), ("poisson_pcg20", M_poisson)]:
            count = [0]
            def callback(_): count[0] += 1
            z, info = cg(A, rhs, x0=np.zeros_like(rhs), rtol=8*np.finfo(float).eps,
                         atol=0.0, maxiter=20, M=M, callback=callback)
            if info < 0 or not np.isfinite(z).all():
                raise FloatingPointError(f"{name} failed on field {i}")
            corrections[name] = z
            executed[name] = count[0]
        np.random.seed(20260912+i)
        corrections["amg"] = np.asarray(vcycle(A, rhs)).reshape(-1)
        if last_vcycle_backend() != "pyamg_smoothed_aggregation":
            raise RuntimeError("actual full AMG required")
        costs.append({"sample": i, **last_vcycle_cost()})
        scores = {"raw": np.abs(rhs), "prediction_magnitude": np.abs(prediction[i]).reshape(-1),
                  "ensemble": sigma[i].reshape(-1), **{key: np.abs(z) for key, z in corrections.items()}}
        # Reference labels enter only after all numerical scores are constructed.
        error = (prediction[i]-truth[i]).reshape(-1)
        direct = spsolve(A, rhs)
        identity = float(np.linalg.norm(direct-error)/np.linalg.norm(error))
        if identity > 1e-8:
            raise AssertionError(f"same-A residual identity failed on field {i}: {identity}")
        identities.append(identity)
        relative_errors.append(float(np.linalg.norm(error)/np.linalg.norm(truth[i])))
        for name, score in scores.items():
            row = {"sample": i, "method": name, **localization_metrics_top10(score, np.abs(error))}
            if name in corrections:
                row["corrected_prediction_relative_l2"] = float(np.linalg.norm(error-corrections[name])/np.linalg.norm(truth[i]))
                row["relative_l2_correction_error"] = float(np.linalg.norm(error-corrections[name])/np.linalg.norm(error))
                row["iterations_executed"] = executed.get(name)
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/"fields.csv", index=False)
    draws = np.random.default_rng(20260912).integers(0,len(a),size=(bootstrap,len(a)))
    methods = {name: {metric: mean_ci(block[metric].to_numpy(), draws) for metric in
                     ["spearman", "auroc", "top10_overlap", "error_capture_top10"]}
               for name, block in frame.groupby("method", sort=False)}
    for name in corrections:
        block = frame[frame.method == name]
        for metric in ["corrected_prediction_relative_l2", "relative_l2_correction_error"]:
            methods[name][metric] = mean_ci(block[metric].to_numpy(), draws)
    wide = frame.pivot(index="sample", columns="method", values="spearman")
    pairs = [("prediction_magnitude","raw"), ("ensemble","raw"), ("diagonal_pcg20","raw"),
             ("poisson_pcg20","raw"), ("amg","raw"), ("poisson_pcg20","amg")]
    return {"methods": methods, "paired_spearman_differences": {
        f"{left}_minus_{right}": mean_ci((wide[left]-wide[right]).to_numpy(), draws) for left,right in pairs},
        "relative_l2_prediction_error": mean_ci(relative_errors, draws),
        "same_A_identity_relative_error_max": max(identities), "amg_cost_per_field": costs}


def plot_summary(evaluation, out):
    import matplotlib.pyplot as plt
    names = ["raw", "prediction_magnitude", "ensemble", "cg20", "diagonal_pcg20", "amg", "poisson_pcg20"]
    labels = ["Raw residual", "Prediction magnitude", "Ensemble spread", "CG20", "Diagonal-PCG20", "AMG cycle", "Poisson-PCG20"]
    entries = [evaluation["methods"][name]["spearman"] for name in names]
    means = np.array([row["mean"] for row in entries])
    errors = np.array([[row["mean"]-row["ci95_low"] for row in entries],
                       [row["ci95_high"]-row["mean"] for row in entries]])
    fig,ax = plt.subplots(figsize=(6.8,3.5),layout="constrained")
    ax.barh(labels,means,xerr=errors,color=["#c65a4a"]+["#3f718c"]*6,capsize=3)
    ax.set_xlabel("Mean per-field Spearman correlation with absolute error")
    ax.set_xlim(0,1.03)
    ax.invert_yaxis()
    ax.set_title("U-Net, fresh fields, N=32, contrast 100")
    ax.spines[["top","right"]].set_visible(False)
    fig.savefig(out/"localization.png",dpi=180)
    fig.savefig(out/"localization.pdf")
    plt.close(fig)


def main(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True, exist_ok=True)
    source_names = ["models/tiny_unet.py", "experiments/exp_unet_robustness.py", "data/generate_contrast.py",
                    "experiments/exp0_pilot_fno.py", "solver/assemble_fd.py", "solver/solve.py",
                    "solver/poisson.py", "solver/multigrid.py", "analysis/metrics.py"]
    report = {"experiment": "unet_fresh_field_robustness", "configuration": {
        key: str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
        "source_sha256": {name: sha256(REPO/name) for name in source_names},
        "versions": {"python": platform.python_version(), "numpy":np.__version__, "scipy":scipy.__version__, "torch":torch.__version__},
        "protocol": {"training_data_seed":0,"test_data_seed":20260912,"grid_N":32,"kappa":100,
                     "selection":"200 fixed epochs by default; no fresh-label tuning, validation selection, or early stopping",
                     "normalization":"coefficient and target statistics fit on training fields only",
                     "architecture":"two-level U-Net, widths16/32/64 by default, GroupNorm/GELU, max-pool, transpose-convolution upsampling, skip concatenation",
                     "optimizer":"Adam, lr0.001 by default, cosine schedule to fixed final epoch",
                     "confidence_intervals":"paired whole-field percentile bootstrap, conditional on these trained networks",
                     "no_state_of_the_art_or_speedup_claim":True},
        "thread_limits":{name:os.environ[name] for name in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"]}}
    start = time.perf_counter()
    a_train, u_train = generate_training(args)
    normalization = fit_normalization(a_train,u_train)
    report["normalization"] = normalization
    report["training_arrays_sha256"] = {"a":array_hash(a_train),"u":array_hash(u_train)}
    if args.pilot_epochs:
        model,history = train_member(0,a_train,u_train,normalization,args,args.pilot_epochs)
        report["pilot_training_history"] = history
        report["parameter_count"] = sum(p.numel() for p in model.parameters())
        report["projected_full_training_s"] = float(np.mean([h["wall_time_s"] for h in history[1:] or history])*args.epochs*len(args.seeds))
        report["pilot_only_no_test_labels_loaded"] = True
        write_json(args.out/"pilot.json",report)
        print(f"PILOT: estimated full training {report['projected_full_training_s']/60:.1f} minutes",flush=True)
        return report
    checkpoints = []
    # Train and freeze every member before loading any fresh reference labels.
    for seed in args.seeds:
        path = args.out/f"unet_k100_seed{seed}.pt"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite trained checkpoint {path}")
        model, history = train_member(seed,a_train,u_train,normalization,args)
        payload = {"model_state_dict":model.state_dict(),"metadata":{
            "seed":seed,"width":args.width,"in_ch":3,"normalization":normalization,
            "training_data_seed":0,"training_count":args.n_train,"epochs":args.epochs,
            "source_sha256":report["source_sha256"],"training_arrays_sha256":report["training_arrays_sha256"]}}
        torch.save(payload,path)
        write_json(path.with_suffix(".json"),payload["metadata"])
        restored = UNet2d(width=args.width,in_ch=3)
        restored.load_state_dict(torch.load(path,map_location="cpu",weights_only=True)["model_state_dict"])
        before = predict_physical(model,a_train[:2],normalization)
        after = predict_physical(restored,a_train[:2],normalization)
        roundtrip = float(np.max(np.abs(before-after)))
        if roundtrip != 0.0:
            raise AssertionError(f"checkpoint roundtrip changed predictions: {roundtrip}")
        checkpoints.append({"seed":seed,"path":str(path.relative_to(REPO)) if path.is_absolute() else str(path),
                            "sha256":sha256(path),"training_history":history,
                            "checkpoint_roundtrip_max_abs":roundtrip})
    report["checkpoints"] = checkpoints
    report["parameter_count"] = sum(p.numel() for p in model.parameters())
    cache = REPO/"results/submission_fresh_32/predictions_k100.npz"
    report["frozen_fno_comparison_cache"] = str(cache.relative_to(REPO))
    report["frozen_fno_comparison_cache_sha256"] = sha256(cache)
    with np.load(cache,allow_pickle=False) as data:
        a,f,truth = [data[name][:args.n_test] for name in ["a","f","u"]]
        signature = json.loads(str(data["signature"].item()))
    if signature["test_seed"] != 20260912 or signature["skip_fields"] != 0 or signature["grid_N"] != 32 or signature["kappa"] != 100:
        raise ValueError("fresh comparison cache has wrong provenance")
    if len(a) != args.n_test:
        raise ValueError("not enough fresh evaluation fields")
    training_set = {array_hash(field) for field in a_train}
    if training_set & {array_hash(field) for field in a}:
        raise AssertionError("training/test coefficient overlap")
    predictions = []
    for checkpoint in checkpoints:
        path = REPO/checkpoint["path"]
        payload = torch.load(path,map_location="cpu",weights_only=True)
        loaded = UNet2d(width=payload["metadata"]["width"],in_ch=3)
        loaded.load_state_dict(payload["model_state_dict"])
        predictions.append(predict_physical(loaded,a,payload["metadata"]["normalization"],args.batch_size))
    members = np.stack(predictions)
    report["evaluation_arrays_sha256"] = {"a":array_hash(a),"f":array_hash(f),"truth":array_hash(truth),"members":array_hash(members)}
    np.savez_compressed(args.out/"predictions_k100.npz",a=a,f=f,u=truth,members=members,
                        signature=json.dumps({"configuration":report["configuration"],"normalization":normalization,"checkpoint_sha256":[p["sha256"] for p in checkpoints]},sort_keys=True))
    report["evaluation"] = evaluate(a,f,truth,members,args.out,args.bootstrap)
    report["evaluation"]["per_member_relative_l2"] = {
        str(seed):float(np.mean(np.linalg.norm((members[j]-truth).reshape(len(a),-1),axis=1)
                               /np.linalg.norm(truth.reshape(len(a),-1),axis=1)))
        for j,seed in enumerate(args.seeds)}
    plot_summary(report["evaluation"],args.out)
    for name,digest in report["source_sha256"].items():
        if sha256(REPO/name) != digest:
            raise RuntimeError(f"source changed during run: {name}")
    if sha256(cache) != report["frozen_fno_comparison_cache_sha256"]:
        raise RuntimeError("FNO comparison cache changed during run")
    report["source_and_comparison_cache_hashes_unchanged"] = True
    report["wall_time_s"] = time.perf_counter()-start
    write_json(args.out/"summary.json",report)
    print("PASS: fixed-epoch training, disjoint fresh-field evaluation and same-A identity complete",flush=True)
    print({name:round(x["spearman"]["mean"],5) for name,x in report["evaluation"]["methods"].items()},flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width",type=int,default=16)
    parser.add_argument("--epochs",type=int,default=200)
    parser.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    parser.add_argument("--lr",type=float,default=.001)
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--n-train",type=int,default=700)
    parser.add_argument("--n-test",type=int,default=200)
    parser.add_argument("--bootstrap",type=int,default=5000)
    parser.add_argument("--pilot-epochs",type=int,default=0)
    parser.add_argument("--out",type=Path,default=REPO/"results/unet_robustness")
    args = parser.parse_args()
    if min(args.width,args.epochs,args.batch_size,args.n_train,args.n_test,args.bootstrap) < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("positive model/training/evaluation settings required")
    if args.n_test < 2 or not 0 <= args.pilot_epochs <= args.epochs or len(set(args.seeds)) != len(args.seeds):
        parser.error("at least two evaluation fields, valid pilot length, and unique seeds required")
    return args


if __name__ == "__main__":
    main(parse_args())
