#!/usr/bin/env python3
"""Normalize legacy per-layer prefill errors to the fused bucket CSV schema."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


PARTS = {
    "qkv_proj": {"q_proj", "k_proj", "v_proj"},
    "o_proj": {"o_proj"},
    "gate_up_proj": {"gate_proj", "up_proj"},
    "down_proj": {"down_proj"},
}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-errors", type=Path, required=True)
    parser.add_argument("--sparse-bf16", type=Path, required=True)
    parser.add_argument("--sparse-nvfp4", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base = read(args.module_errors)
    overrides = {
        "sparse_bf16": read(args.sparse_bf16),
        "sparse_nvfp4": read(args.sparse_nvfp4),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for method in ("dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "w4a16_ours"):
        source_method = "dense_nvfp4" if method == "w4a16_ours" else method
        source = overrides.get(source_method, base)
        rows = []
        for bucket in range((args.layers + 7) // 8):
            for fused, parts in PARTS.items():
                total = 0.0
                for layer in range(bucket * 8, min((bucket + 1) * 8, args.layers)):
                    values = [float(row["local_rel_mse"]) for row in source
                              if int(row["layer"]) == layer and
                              row["module_type"] in parts and
                              row["method"] == source_method]
                    if not values:
                        raise RuntimeError(f"missing layer={layer}, type={fused}, method={source_method}")
                    total += sum(values) / len(values)
                rows.append({"phase": "prefill", "method": method,
                             "layer_bucket": bucket, "fused_type": fused,
                             "output_rel_mse": total,
                             "backend": "normalized_legacy_canonical_feature"})
        output = args.output_dir / f"prefill_{method}.csv"
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
