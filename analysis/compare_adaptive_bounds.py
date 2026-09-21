"""Paired stopping-work comparison for identical cached prediction fields."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compare(baseline_path: Path, flux_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite comparison: {output}")
    baseline, flux = [json.loads(path.read_text()) for path in (baseline_path, flux_path)]
    for name in ("q", "target_certified_high_recall", "check_every", "maxiter"):
        if baseline["protocol"][name] != flux["protocol"][name]:
            raise ValueError(f"stopping protocol differs: {name}")
    if baseline["protocol"].get("bound_kind", "poisson") != "poisson" or flux["protocol"]["bound_kind"] != "flux":
        raise ValueError("expected Poisson baseline and flux comparison")
    result = {"bootstrap_replicates": 5000, "bootstrap_unit": "paired coefficient field",
              "baseline_summary_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
              "flux_summary_sha256": hashlib.sha256(flux_path.read_bytes()).hexdigest(),
              "results": {}}
    table = []
    for index, (key, new) in enumerate(flux["results"].items()):
        old = baseline["results"][key]
        if old["cache_sha256"] != new["cache_sha256"] or old["n_fields"] != new["n_fields"]:
            raise ValueError(f"comparison fields differ for {key}")
        frames = [pd.read_csv(path.parent / f"fields_{key}.csv").sort_values("sample")
                  for path in (baseline_path, flux_path)]
        if any(frame["sample"].tolist() != list(range(new["n_fields"])) for frame in frames):
            raise ValueError("duplicate, missing, or differently ordered sample indices")
        n = new["n_fields"]
        draws = np.random.default_rng(20260912 + index).integers(0, n, size=(5000, n))
        paired = {}
        for metric in ("iterations", "bound_poisson_solves", "fft_transforms",
                       "certified_high_recall", "relative_l2_correction_error"):
            difference = frames[1][metric].to_numpy() - frames[0][metric].to_numpy()
            low, high = np.quantile(difference[draws].mean(axis=1), [0.025, 0.975])
            paired[metric] = {"mean_flux_minus_poisson": float(difference.mean()),
                              "ci95_low": float(low), "ci95_high": float(high)}
        delta = frames[1].iterations.to_numpy() - frames[0].iterations.to_numpy()
        result["results"][key] = {
            "n_fields": n, "grid_N": new["grid_N"], "kappa": new["kappa"],
            "baseline_mean_iterations": float(frames[0].iterations.mean()),
            "flux_mean_iterations": float(frames[1].iterations.mean()),
            "fraction_stopping_earlier": float(np.mean(delta < 0)),
            "fraction_stopping_later": float(np.mean(delta > 0)),
            "baseline_target_met_fraction": float(frames[0].target_met.mean()),
            "flux_target_met_fraction": float(frames[1].target_met.mean()),
            "flux_relative_l2_correction_error": float(frames[1].relative_l2_correction_error.mean()),
            "flux_false_high_total": int(frames[1].false_high.sum()),
            "flux_false_low_total": int(frames[1].false_low.sum()),
            "paired_differences": paired,
        }
        table.append({"condition": key, **{name: value for name, value in result["results"][key].items()
                                           if name != "paired_differences"},
                      **paired["iterations"]})
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(table).to_csv(output.with_suffix(".csv"), index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--flux", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline, args.flux, args.out)
    for name, block in result["results"].items():
        delta = block["paired_differences"]["iterations"]
        print(f"{name}: {block['baseline_mean_iterations']:.3f} -> {block['flux_mean_iterations']:.3f}, "
              f"paired delta {delta['mean_flux_minus_poisson']:.3f} "
              f"[{delta['ci95_low']:.3f}, {delta['ci95_high']:.3f}], "
              f"later={block['fraction_stopping_later']:.3f}")


if __name__ == "__main__":
    main()
