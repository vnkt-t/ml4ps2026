"""Compare all stored adaptive fields and stopping histories without tolerances."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / "results/adaptive_rank_stopping"
REPLAY = ROOT / "results/replay_20260912/adaptive_rank_stopping"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def structural_differences(left, right, path="", output=None):
    if output is None:
        output=[]
    if isinstance(left,dict) and isinstance(right,dict):
        for key in sorted(set(left)|set(right)):
            loc=f"{path}.{key}" if path else key
            if key not in left or key not in right:
                output.append({"path":loc,"original":left.get(key),"replay":right.get(key),"kind":"missing_key"})
            else:
                structural_differences(left[key],right[key],loc,output)
    elif isinstance(left,list) and isinstance(right,list):
        if len(left)!=len(right):
            output.append({"path":path,"original_length":len(left),"replay_length":len(right),"kind":"length"})
        for i,(a,b) in enumerate(zip(left,right)):
            structural_differences(a,b,f"{path}[{i}]",output)
    elif left != right:
        row={"path":path,"original":left,"replay":right,"kind":"value"}
        if isinstance(left,(int,float)) and isinstance(right,(int,float)):
            row["absolute_difference"]=abs(float(left)-float(right))
        output.append(row)
    return output


def main():
    snapshot=Path(__file__).resolve().parent
    manifest=json.loads((snapshot/"manifest.json").read_text())
    stable={name:sha(ROOT/name)==expected for name,expected in {**manifest["sources"],**manifest["inputs"]}.items()}
    snapshots={name:sha(snapshot/name)==expected for name,expected in manifest["sources"].items()}
    report={"experiment":"adaptive_current_source_replay_comparison",
            "original_directory":str(ORIGINAL.relative_to(ROOT)),"replay_directory":str(REPLAY.relative_to(ROOT)),
            "source_snapshot_directory":str(snapshot.relative_to(ROOT)),"command":manifest["command"],
            "comparison":"all CSV cells and complete history trees; exact numeric equality, no tolerance/filtering",
            "input_and_current_sources_unchanged":all(stable.values()),"stability_checks":stable,
            "source_snapshots_byte_identical":all(snapshots.values()),"snapshot_checks":snapshots,
            "conditions":{},"output_sha256":{},"numeric_cell_differences_by_column":{},
            "history_leaf_difference_count":0,"summary_metric_leaf_differences":[]}
    totals=Counter()
    originals=sorted(ORIGINAL.glob("fields_*.csv")); replays=sorted(REPLAY.glob("fields_*.csv"))
    report["field_files_match"]=[p.name for p in originals]==[p.name for p in replays]
    if not report["field_files_match"]:
        raise AssertionError("original and replay field-file sets differ")
    for old_path in originals:
        key=old_path.stem.removeprefix("fields_")
        new_path=REPLAY/old_path.name
        old,new=pd.read_csv(old_path),pd.read_csv(new_path)
        if list(old.columns)!=list(new.columns) or len(old)!=len(new):
            raise AssertionError(f"CSV schema/length differs: {key}")
        if not old["sample"].equals(new["sample"]):
            raise AssertionError(f"sample order differs: {key}")
        columns={}
        for column in old.columns:
            a,b=old[column],new[column]
            equal=(a==b)|(a.isna()&b.isna())
            count=int((~equal).sum())
            numeric=pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)
            row={"different_cells":count,"numeric":numeric}
            if numeric:
                delta=b.to_numpy(dtype=float)-a.to_numpy(dtype=float)
                row.update({"max_absolute_difference":float(np.nanmax(np.abs(delta))),
                            "original_sum":float(a.sum()),"replay_sum":float(b.sum())})
                totals[column]+=count
                if count:
                    changed=np.flatnonzero(~equal.to_numpy())
                    row["examples"]=[{"sample":int(old.iloc[i]["sample"]),"original":float(a.iloc[i]),
                                      "replay":float(b.iloc[i]),"delta":float(delta[i])} for i in changed[:5]]
                    row["delta_min"]=float(delta.min());row["delta_max"]=float(delta.max())
            elif count:
                row["examples"]=[{"sample":int(old.iloc[i]["sample"]),"original":str(a.iloc[i]),
                                  "replay":str(b.iloc[i])} for i in np.flatnonzero(~equal.to_numpy())[:5]]
            columns[column]=row
        old_history=ORIGINAL/f"histories_{key}.json";new_history=REPLAY/old_history.name
        hist_diff=structural_differences(json.loads(old_history.read_text()),json.loads(new_history.read_text()))
        report["history_leaf_difference_count"]+=len(hist_diff)
        report["conditions"][key]={"fields":len(old),"columns":columns,
            "history_exactly_equal":not hist_diff,"history_difference_count":len(hist_diff),
            "history_differences":hist_diff,
            "current_false_high_total":int(new.false_high.sum()),"current_false_low_total":int(new.false_low.sum())}
        for path in [new_path,new_history]:report["output_sha256"][str(path.relative_to(ROOT))]=sha(path)
    old_summary=json.loads((ORIGINAL/"summary.json").read_text())
    new_summary=json.loads((REPLAY/"summary.json").read_text())
    for key in old_summary["results"]:
        report["summary_metric_leaf_differences"]+=structural_differences(
            old_summary["results"][key]["metrics"],new_summary["results"][key]["metrics"],key)
    report["summary_protocol_differences"]=structural_differences(old_summary["protocol"],new_summary["protocol"])
    report["summary_source_hash_differences"]=structural_differences(old_summary["source_hashes"],new_summary["source_hashes"])
    report["numeric_cell_differences_by_column"]=dict(totals)
    report["n_conditions"]=len(originals)
    report["n_fields"]=sum(x["fields"] for x in report["conditions"].values())
    report["all_stopping_histories_exactly_reproduced"]=report["history_leaf_difference_count"]==0
    report["all_non_fft_csv_values_exactly_reproduced"]=all(
        values["different_cells"]==0 for item in report["conditions"].values()
        for column,values in item["columns"].items() if column!="fft_transforms")
    report["fft_counter_note"]="Any fft_transforms differences are retained explicitly above. Current convention counts forward/inverse 2D DST calls: 2*(preconditioner_applications+bound_poisson_solves). No numerical differences are discarded."
    report["output_sha256"][str((REPLAY/"summary.json").relative_to(ROOT))]=sha(REPLAY/"summary.json")
    report["comparison_source_sha256"]=sha(Path(__file__))
    output=ROOT/"results/adaptive_current_replay_comparison.json"
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({key:report[key] for key in ["n_conditions","n_fields","input_and_current_sources_unchanged",
        "source_snapshots_byte_identical","all_stopping_histories_exactly_reproduced",
        "all_non_fft_csv_values_exactly_reproduced","numeric_cell_differences_by_column"]},indent=2))
    print("Summary metric differences:",len(report["summary_metric_leaf_differences"]))
    return report


if __name__=="__main__":main()
