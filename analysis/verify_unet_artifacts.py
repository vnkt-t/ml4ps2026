"""Independently reload every U-Net member and verify saved fresh predictions.

No training and no metric selection occur. The original run remains immutable;
the check writes a separate verification report beside it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[name] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
import torch

from models.tiny_unet import UNet2d
from experiments.exp0_pilot_fno import build_inputs


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(directory):
    torch.set_num_threads(1)
    summary_path = directory/"summary.json"
    prediction_path = directory/"predictions_k100.npz"
    summary = json.loads(summary_path.read_text())
    hashes = {str(p.relative_to(REPO)):digest(p) for p in [summary_path,prediction_path]}
    with np.load(prediction_path,allow_pickle=False) as data:
        a,f,u,members = [data[name] for name in ["a","f","u","members"]]
    with np.load(REPO/summary["frozen_fno_comparison_cache"],allow_pickle=False) as data:
        for name,saved in [("a",a),("f",f),("u",u)]:
            if not np.array_equal(saved,data[name][:len(a)]):
                raise AssertionError(f"saved U-Net {name} fields differ from frozen FNO evaluation cache")
    reference_normalization = json.loads((REPO/"results/checkpoints/fno_kappa100_seed0.json").read_text())["normalization"]
    checks = []
    for j,row in enumerate(summary["checkpoints"]):
        path = REPO/row["path"]
        if digest(path) != row["sha256"]:
            raise AssertionError(f"checkpoint hash mismatch: {path}")
        hashes[str(path.relative_to(REPO))] = digest(path)
        payload = torch.load(path,map_location="cpu",weights_only=True)
        meta = payload["metadata"]
        if meta["seed"] != summary["configuration"]["seeds"][j] or meta["epochs"] != summary["configuration"]["epochs"]:
            raise AssertionError("checkpoint seed or fixed epoch count mismatch")
        norm = meta["normalization"]
        if norm != summary["normalization"] or norm != reference_normalization:
            raise AssertionError("U-Net and FNO training normalizations differ")
        model = UNet2d(width=meta["width"],in_ch=meta["in_ch"])
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        inputs = build_inputs(a,norm["logmean"],norm["logstd"],a.shape[1])
        blocks = []
        # Reimplement the physical prediction path rather than calling the
        # evaluator's helper. Keep its batch size to test exact serialization.
        batch = summary["configuration"]["batch_size"]
        with torch.inference_mode():
            for start in range(0,len(a),batch):
                normalized = model(torch.as_tensor(inputs[start:start+batch])).numpy().astype(np.float64)
                blocks.append(normalized*norm["ustd"]+norm["umean"])
        restored = np.concatenate(blocks)
        maximum = float(np.max(np.abs(restored-members[j])))
        if maximum != 0.0:
            raise AssertionError(f"checkpoint seed {meta['seed']} reload changed predictions by {maximum}")
        checks.append({"seed":meta["seed"],"max_abs_prediction_difference":maximum,
                       "fixed_epochs":meta["epochs"],"fno_training_normalization_exact_match":True})
    for name,expected in hashes.items():
        if digest(REPO/name) != expected:
            raise AssertionError(f"input changed during verification: {name}")
    report = {"experiment":"unet_checkpoint_prediction_verification","checks":checks,
              "fresh_fields_match_fno_cache_exactly":True,"n_fields":len(a),
              "input_sha256":hashes,"verification_source_sha256":digest(Path(__file__)),
              "scope":"exact same-environment checkpoint/prediction consistency, not a retraining or generalization guarantee"}
    (directory/"artifact_verification.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(f"PASS: all {len(checks)} checkpoints reproduce every saved prediction exactly; all four training-normalization constants match FNO",flush=True)
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,default=REPO/"results/unet_robustness")
    args=parser.parse_args()
    verify(args.directory.resolve())
