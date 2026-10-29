#!/usr/bin/env python3
"""Combine calibrated quality coefficients and kernel actions for the DP solver."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from daso.modeling.quality import FUSED_TYPES
from daso.policy import METHODS


KERNEL = {"dense_bf16": "dense_bf16", "dense_nvfp4": "dense_nvfp4",
          "sparse_bf16": "sparse_bf16", "sparse_nvfp4": "sparse_nvfp4",
          "w4a16_ours": "marlin_nvfp4"}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-model", type=Path, required=True)
    parser.add_argument("--local-errors", type=Path, required=True)
    parser.add_argument("--latency-actions", type=Path, required=True)
    parser.add_argument("--scenario", choices=("prefill_only", "prefill_decode"), required=True)
    parser.add_argument("--decode-steps", type=int, default=63)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = json.loads(args.quality_model.read_text())
    coefficients = model["coefficients"]
    scale = float(model["feature_scale"])
    quality_intercept = float(model["bias"])
    local = {}
    phases = ("prefill",) if args.scenario == "prefill_only" else ("prefill", "decode")
    for phase in phases:
        for method in METHODS:
            if method == "dense_bf16":
                continue
            for row in read(args.local_errors / f"{phase}_{method}.csv"):
                local[phase, int(row["layer_bucket"]), row["fused_type"], method] = \
                    float(row["output_rel_mse"])
    latency = {(row["phase"], row["module_name"], row["kernel"]): row
               for row in read(args.latency_actions)}
    module_names = sorted({key[1] for key in latency})
    rows = []
    for name in module_names:
        layer = int(name.split(".")[2]); bucket = layer // 8
        kind = name.rsplit(".", 1)[-1]
        if kind not in FUSED_TYPES:
            continue
        pre_methods = METHODS
        decode_methods = METHODS
        for pre in pre_methods:
            for decode in decode_methods:
                if args.scenario == "prefill_only" and decode != pre:
                    continue
                pre_row = latency.get(("prefill", name, KERNEL[pre]))
                decode_row = latency.get(("decode", name, KERNEL[decode])) if args.scenario == "prefill_decode" else None
                supported = pre_row is not None and pre_row.get("supported", "True").lower() == "true"
                supported &= decode_row is None or decode_row.get("supported", "True").lower() == "true"
                if not supported:
                    continue
                quality = 0.0
                for phase_index, (phase, method) in enumerate((("prefill", pre), ("decode", decode))):
                    if args.scenario == "prefill_only" and phase == "decode":
                        continue
                    error = 0.0 if method == "dense_bf16" else local[phase, bucket, kind, method]
                    quality += error * float(coefficients[phase_index][METHODS.index(method)][bucket][FUSED_TYPES.index(kind)]) / scale
                latency_ms = float(pre_row["latency_ms"])
                if decode_row is not None:
                    latency_ms += args.decode_steps * float(decode_row["latency_ms"])
                rows.append({"module_name": name, "prefill_method": pre,
                             "decode_method": decode, "quality_cost": quality,
                             "quality_intercept": quality_intercept,
                             "latency_ms": latency_ms, "supported": True})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
