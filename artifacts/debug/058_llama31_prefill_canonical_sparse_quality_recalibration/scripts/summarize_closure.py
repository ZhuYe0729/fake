#!/usr/bin/env python3
"""Join solved-policy predictions with measured canonical prefill closure data."""
from __future__ import annotations
import csv
import json
import statistics
from scenario import EXP

def main() -> None:
    predicted = list(csv.DictReader((EXP / "pareto/predicted_points.csv").open()))
    bridge_path = EXP / "pareto/dense_nvfp4_bridge.csv"
    if bridge_path.exists():
        for bridge in csv.DictReader(bridge_path.open()):
            predicted.append({
                "point_index": "",
                "policy_id": bridge["policy_id"],
                "quality_budget": "bridge_dense_nvfp4",
                "predicted_delta_nll": bridge["predicted_delta_nll"],
                "raw_predicted_linear_ms": bridge["raw_predicted_linear_ms"],
                "corrected_e2e_prediction_ms": "",
                "count_dense_bf16": str(128 - int(bridge["dense_nvfp4_modules"])),
                "count_dense_nvfp4": bridge["dense_nvfp4_modules"],
                "count_sparse_bf16": "0",
                "count_sparse_nvfp4": "0",
                "count_w4a16_ours": "0",
            })
    base = json.loads((EXP / "nll/raw/p00.json").read_text())["avg_nll"]
    rows = []
    for item in predicted:
        policy = item["policy_id"]; nll_path = EXP / "pareto/closure/nll" / f"{policy}.json"; run_dir = EXP / "pareto/closure/speed" / policy / "runs"
        if not nll_path.exists() or not all((run_dir / f"measured_{i}.json").exists() for i in range(5)): continue
        nll = json.loads(nll_path.read_text()); values = [json.loads((run_dir / f"measured_{i}.json").read_text())["elapsed_ms"] for i in range(5)]
        rows.append({**item, "measured_e2e_ms": statistics.median(values), "measured_runs_ms": json.dumps(values), "measured_delta_nll": float(nll["avg_nll"]) - float(base)})
    if not rows: raise RuntimeError("no completed closure points")
    output = EXP / "pareto/closure_summary.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(output)
if __name__ == "__main__": main()
