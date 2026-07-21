#!/usr/bin/env python3
"""Project legal phase actions and select fixed roofline speed-calibration policies."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "artifacts/debug/055_llama2_prefill_decode_canonical_pareto/llama2_7b_chat"
KERNEL = {"dense_bf16": "dense_bf16", "dense_nvfp4": "dense_nvfp4", "sparse_bf16": "sparse_bf16",
          "sparse_nvfp4": "sparse_nvfp4", "w4a16_ours": "marlin_nvfp4"}
# First seven fit the calibrator; last five are held out.  They are selected
# from roofline-predicted cost coverage before collecting E2E timings.
SELECTED = ("p00", "p01", "p02", "p03", "p04", "p07", "p21",
            "p37", "p05", "p24", "p18", "p23")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    actions = {(row["phase"], row["module_name"], row["kernel"]): row
               for row in read(EXP / "speed/action_support.csv")}
    labels = {row["policy_id"]: row for row in read(EXP / "nll/prefill_decode.csv")}
    output = EXP / "speed/calibration"; policy_dir = output / "policies"; policy_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for policy_id in SELECTED:
        policy = json.loads((EXP / "policies/prefill_decode" / f"{policy_id}.json").read_text())
        repaired = 0
        for name, item in policy["method_map"].items():
            for phase in ("prefill", "decode"):
                method = item[f"{phase}_method"]
                if actions[phase, name, KERNEL[method]]["supported"] != "True":
                    item[f"{phase}_method"] = "dense_nvfp4"; repaired += 1
        policy["policy_kind"] = "speed_calibration_legal_projection"
        legal = policy_dir / f"{policy_id}.json"; legal.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
        raw_pre = sum(float(actions["prefill", name, KERNEL[item["prefill_method"]]]["latency_ms"])
                      for name, item in policy["method_map"].items())
        raw_dec = sum(float(actions["decode", name, KERNEL[item["decode_method"]]]["latency_ms"])
                      for name, item in policy["method_map"].items())
        rows.append({"policy_id": policy_id, "policy_json": str(legal), "repaired_unsupported_actions": repaired,
                     "raw_predicted_prefill_ms": raw_pre, "raw_predicted_decode_ms": raw_dec,
                     "raw_predicted_linear_ms": raw_pre + 80 * raw_dec,
                     "measured_delta_nll_original_policy": labels[policy_id]["target_delta_nll"]})
    with (output / "design.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output / "metadata.json").write_text(json.dumps({"scenario": "prefill_decode", "selection": list(SELECTED),
        "train_policies": list(SELECTED[:7]), "holdout_policies": list(SELECTED[7:]),
        "raw_cost": "sum of KernelLatencyPredictor phase latencies, Mpre=32768 + 80*Mdecode=16; conversion excluded", 
        "legal_projection": "only actions marked unsupported by the kernel predictor are replaced with dense_nvfp4",
        "runner": "five reused-LLM phase-heterogeneous vLLM samples per policy, b=16 input=2048 output=80"}, indent=2) + "\n")
    print(output / "design.csv")


if __name__ == "__main__":
    main()
