#!/usr/bin/env python3
"""Build the fixed 12-policy E2E speed calibration design without NLL labels."""
from __future__ import annotations
import csv
import json
from scenario import EXP, SOURCE_038, KERNEL

SELECTED = ("p00", "p01", "p02", "p03", "p04", "p37", "p39", "p42", "p45", "p52", "p60", "p68")

def main() -> None:
    actions = list(csv.DictReader((SOURCE_038 / "action_support.csv").open()))
    latency = {(r["module_name"], r["kernel"]): float(r["latency_ms"])
               for r in actions if r["supported"] == "True"}
    rows = []
    for policy_id in SELECTED:
        policy_path = EXP / "policies/prefill_only" / f"{policy_id}.json"
        policy = json.loads(policy_path.read_text())
        mapping = policy["method_map"]
        raw = sum(latency[name, KERNEL[item["prefill_method"]]] for name, item in mapping.items())
        rows.append({"policy_id": policy_id, "policy_json": str(policy_path),
                     "raw_predicted_linear_ms": raw,
                     **{f"count_{method}": sum(v["prefill_method"] == method for v in mapping.values())
                        for method in KERNEL}})
    output = EXP / "speed/calibration"; output.mkdir(parents=True, exist_ok=True)
    with (output / "design.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    metadata = {"selected": list(SELECTED), "train_policies": list(SELECTED[:7]),
                "holdout_policies": list(SELECTED[7:]),
                "kernel_asset": str(SOURCE_038 / "action_support.csv"),
                "raw_prediction": "sum of Llama3 B=8/S=2048 KernelLatencyPredictor module latencies",
                "measurement": "canonical phase runtime; fresh loaded-vLLM process; warmup plus five samples"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(output / "design.csv")
if __name__ == "__main__": main()
