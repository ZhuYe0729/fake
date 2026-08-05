#!/usr/bin/env python3
"""Generate quality-ranked dense-NVFP4 bridges for the mid-speed gap."""
from __future__ import annotations

import csv
import json

from scenario import EXP, KERNEL, SOURCE_038, SOURCE_057, SOURCE_058, TYPES


COUNTS = (96, 112)


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    model = json.loads((SOURCE_058 / "reports/quality/model.json").read_text())
    scale = float(model["feature_scale"])

    def coefficient(bucket, fused_type):
        raw = (
            float(model["global"])
            + float(model["method"][1])
            + float(model["bucket"][bucket])
            + float(model["type"][TYPES.index(fused_type)])
        )
        return max(0.0, raw) / scale

    local_errors = {
        (int(row["layer_bucket"]), row["fused_type"]): float(row["output_rel_mse"])
        for row in read(SOURCE_057 / "local_errors/prefill_dense_nvfp4.csv")
    }
    latency = {
        (row["module_name"], row["kernel"]): float(row["latency_ms"])
        for row in read(SOURCE_038 / "action_support.csv")
        if row["supported"] == "True"
    }
    candidates = []
    for layer in range(32):
        for group, fused_type in (
            ("self_attn", "qkv_proj"),
            ("self_attn", "o_proj"),
            ("mlp", "gate_up_proj"),
            ("mlp", "down_proj"),
        ):
            name = f"model.layers.{layer}.{group}.{fused_type}"
            bucket = layer // 8
            penalty = local_errors[bucket, fused_type] * coefficient(bucket, fused_type)
            saving = latency[name, KERNEL["dense_bf16"]] - latency[name, KERNEL["dense_nvfp4"]]
            candidates.append((penalty / max(saving, 1e-9), name, penalty, saving))
    candidates.sort()

    output_rows = []
    policy_dir = EXP / "policies"
    policy_dir.mkdir(parents=True, exist_ok=True)
    for count in COUNTS:
        selected = {name for _, name, _, _ in candidates[:count]}
        policy_id = f"bridge_dense_nvfp4_{count:03d}"
        mapping = {
            name: {
                "prefill_method": "dense_nvfp4" if name in selected else "dense_bf16",
                "decode_method": "dense_nvfp4" if name in selected else "dense_bf16",
            }
            for _, name, _, _ in candidates
        }
        policy = {
            "policy_id": policy_id,
            "scenario": "prefill_only",
            "policy_kind": "dense_nvfp4_bridge",
            "default_prefill_method": "dense_bf16",
            "default_decode_method": "dense_bf16",
            "modules_to_not_convert": ["lm_head"],
            "method_map": mapping,
        }
        (policy_dir / f"{policy_id}.json").write_text(
            json.dumps(policy, indent=2, sort_keys=True) + "\n"
        )
        output_rows.append(
            {
                "policy_id": policy_id,
                "dense_nvfp4_modules": count,
                "predicted_delta_nll": sum(item[2] for item in candidates[:count]),
                "raw_predicted_linear_ms": sum(
                    latency[
                        name,
                        KERNEL["dense_nvfp4"] if name in selected else KERNEL["dense_bf16"],
                    ]
                    for _, name, _, _ in candidates
                ),
            }
        )

    output = EXP / "midrange_bridge_candidates.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(output)


if __name__ == "__main__":
    main()
