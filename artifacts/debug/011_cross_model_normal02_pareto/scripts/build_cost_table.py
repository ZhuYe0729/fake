#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common_pareto import (
    DEBUG_ROOT,
    METHODS,
    SCENARIO,
    f,
    load_latency_from_pred_candidates,
    load_module_quality_rows,
    normalize_group_name,
    quality_cost,
    write_csv,
    write_json,
)


FORMULAS = (
    "local_rel_mse",
    "local_rel_mse_log_numel",
    "local_rel_mse_log_numel_layer_family",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cross-model normal_02 quality/latency candidate table.")
    parser.add_argument("--model", choices=["llama2-7b", "llama31-8b", "qwen35-9b"], default="llama31-8b")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--quality-formula", choices=FORMULAS, default="local_rel_mse_log_numel_layer_family")
    parser.add_argument("--quality-path", type=Path, default=None)
    parser.add_argument("--latency-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_root = args.output_root if args.model == "llama2-7b" else args.output_root / "models" / args.model
    quality_path = args.quality_path or model_root / "sensitivity" / "module_method_errors.csv"
    latency_path = args.latency_path or args.output_root / "latency_pred" / "pred" / "normal_02" / f"{args.model}_pred_candidates.csv"
    quality_rows = load_module_quality_rows(quality_path)
    latency_lookup = load_latency_from_pred_candidates(latency_path)

    candidates: list[dict[str, Any]] = []
    missing_latency: list[dict[str, Any]] = []
    for qrow in quality_rows:
        method = qrow.get("method", "")
        if method not in METHODS:
            continue
        module_name = str(qrow["module_name"])
        group = normalize_group_name(module_name)
        n_val = int(f(qrow, "out_features"))
        k_val = int(f(qrow, "in_features"))
        key = (group, n_val, k_val, str(method))
        lat = latency_lookup.get(key)
        if lat is None:
            missing_latency.append(
                {"method": method, "module_name": module_name, "linear_group": group, "n": n_val, "k": k_val}
            )
            continue
        item = dict(qrow)
        item.update(
            {
                "quality_formula": args.quality_formula,
                "quality_cost": quality_cost(qrow, args.quality_formula),
                "latency_source": str(latency_path),
                "linear_group": group,
                "prefill_ms": lat["prefill_ms"],
                "decode_ms": lat["decode_ms"],
                "conversion_ms": lat["conversion_ms"],
                "latency_cost": lat["total_ms"],
                "prefill_backend": lat["prefill_backend"],
                "decode_backend": lat["decode_backend"],
                "output_tokens": SCENARIO["output_tokens"],
                "supported": str(lat["supported"]),
                "unsupported_reason": lat["reason"] if not lat["supported"] else "",
            }
        )
        candidates.append(item)

    dense_latency = {row["module_name"]: f(row, "latency_cost") for row in candidates if row["method"] == "dense_bf16"}
    for row in candidates:
        row["latency_gain_vs_dense"] = dense_latency.get(row["module_name"], f(row, "latency_cost")) - f(row, "latency_cost")
        row["quality_delta_vs_dense"] = f(row, "quality_cost")

    out_root = model_root / "costs"
    write_csv(out_root / "module_method_candidates.csv", candidates)
    write_csv(out_root / "missing_latency.csv", missing_latency)
    write_json(
        out_root / "build_cost_table_metadata.json",
        {
            "model": args.model,
            "scenario": SCENARIO,
            "quality_path": str(quality_path),
            "latency_path": str(latency_path),
            "candidate_rows": len(candidates),
            "missing_latency_rows": len(missing_latency),
            "quality_formula": args.quality_formula,
        },
    )
    print(f"wrote {len(candidates)} candidate rows to {out_root / 'module_method_candidates.csv'}")
    print(f"missing latency rows: {len(missing_latency)}")


if __name__ == "__main__":
    main()
