"""Compare a complete fresh-field checkpoint replay against retained evidence.

Timing/provenance text is intentionally not compared as scientific output.
All cached physical arrays, member predictions and per-field diagnostic values
are compared. The allowed metadata difference is the documented generator edit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(replay_root):
    report = {"experiment": "complete_frozen_checkpoint_replay_comparison",
              "status": "PASS", "n_fields": 0, "per_configuration": {},
              "scope": "Regenerated fresh coefficients, forcing, truth and all frozen-member predictions; all retained per-field scores and correction metrics. Wall times are not reproducibility targets.",
              "source_sha256": {str(p.relative_to(REPO)): sha(p) for p in [
                  Path(__file__), REPO/"experiments/exp_submission_validation.py",
                  REPO/"data/generate_contrast.py", REPO/"experiments/exp_real_fno_multi_kappa.py"]}}
    failures = []
    for n in [32, 64]:
        original = REPO/f"results/submission_fresh_{n}"
        replay = replay_root/f"submission_fresh_{n}"
        for kappa in [10, 100, 500]:
            left = original/f"predictions_k{kappa}.npz"
            right = replay/f"predictions_k{kappa}.npz"
            item = {"original_cache_sha256": sha(left), "replay_cache_sha256": sha(right),
                    "arrays": {}, "tables": {}}
            with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
                for name in ["a", "f", "u", "members"]:
                    same = np.array_equal(a[name], b[name])
                    item["arrays"][name] = {"bitwise_equal": same,
                        "max_absolute_difference": float(np.max(np.abs(a[name]-b[name])))}
                    if not same: failures.append(f"N{n}/k{kappa}/{name}: arrays differ")
                sa, sb = [json.loads(str(x["signature"].item())) for x in [a,b]]
                changed = [key for key in set(sa)|set(sb) if sa.get(key) != sb.get(key)]
                item["signature_fields_changed"] = sorted(changed)
                if set(changed)-{"generator_sha256"}: failures.append(f"unexpected signature drift: {changed}")
                if "generator_sha256" in changed:
                    note = json.loads((REPO/"reproducibility/generator_documentation_update.json").read_text())
                    if sa["generator_sha256"] != note["original_sha256"] or sb["generator_sha256"] != note["current_sha256"]:
                        failures.append("generator signature drift does not match documented source correction")
                report["n_fields"] += int(a["a"].shape[0])
            for stem in ["metrics", "norms"]:
                pa, pb = [directory/f"{stem}_k{kappa}.csv" for directory in [original,replay]]
                a, b = pd.read_csv(pa), pd.read_csv(pb)
                if list(a.columns) != list(b.columns) or a.shape != b.shape:
                    raise AssertionError(f"different table layout: {pa}, {pb}")
                worst = 0.
                for column in a.columns:
                    if pd.api.types.is_numeric_dtype(a[column]):
                        aa, bb = a[column].to_numpy(), b[column].to_numpy()
                        if np.isinf(aa).any() or np.isinf(bb).any():
                            failures.append(f"nonfinite metric in {pa}/{column}")
                        mask = np.isfinite(aa) & np.isfinite(bb)
                        if not np.array_equal(np.isnan(aa), np.isnan(bb)):
                            failures.append(f"missing-value mismatch: {pa}/{column}")
                        if not np.allclose(aa[mask],bb[mask],atol=2e-12,rtol=2e-10):
                            failures.append(f"metric mismatch: {pa}/{column}")
                        if mask.any(): worst = max(worst, float(np.max(np.abs(aa[mask]-bb[mask]))))
                    elif not a[column].equals(b[column]):
                        failures.append(f"row-label mismatch: {pa}/{column}")
                item["tables"][stem] = {"rows": len(a), "columns": len(a.columns),
                    "max_absolute_difference": worst, "original_sha256":sha(pa), "replay_sha256":sha(pb)}
            report["per_configuration"][f"N{n}_k{kappa}"] = item
    report["failures"] = failures
    report["status"] = "FAIL" if failures else "PASS"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, default=REPO/"results/replay_20260912")
    parser.add_argument("--out", type=Path, default=REPO/"results/submission_replay_comparison.json")
    args = parser.parse_args()
    report = compare(args.replay_root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(f"{report['status']}: {report['n_fields']} fresh fields; {len(report['failures'])} mismatches")
    if report["failures"]: raise SystemExit(1)


if __name__ == "__main__":
    main()
