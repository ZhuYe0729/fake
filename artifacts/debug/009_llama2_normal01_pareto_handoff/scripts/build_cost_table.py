#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common_pareto import (
    DEBUG_ROOT,
    METHODS,
    POLICY_JSON_PATH,
    f,
    get_latency_for_module,
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
    "local_rel_mse_log_numel_activation_outlier",
    "local_rel_mse_log_numel_weight_outlier",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-module method quality/latency candidate table.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--quality-formula", choices=FORMULAS, default="local_rel_mse_log_numel_layer_family")
    parser.add_argument("--policy-json", type=Path, default=POLICY_JSON_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quality_rows = load_module_quality_rows()
    latency_lookup = load_latency_from_policy_json(args.policy_json)
    output_tokens = 32

    candidates: list[dict[str, Any]] = []
    missing_latency: list[dict[str, Any]] = []
    for qrow in quality_rows:
        method = qrow.get("method", "")
        if method not in METHODS:
            continue
        module_name = str(qrow["module_name"])
        group = normalize_group_name(module_name)
        try:
            lat = get_latency_for_module(module_name, method, latency_lookup, output_tokens)
        except KeyError:
            n_val = int(f(qrow, "out_features"))
            k_val = int(f(qrow, "in_features"))
            missing_latency.append(
                {"method": method, "module_name": module_name, "linear_group": group, "n": n_val, "k": k_val}
            )
            continue

        item = dict(qrow)
        item.update(
            {
                "quality_formula": args.quality_formula,
                "quality_cost": quality_cost(qrow, args.quality_formula),
                "latency_source": "policy_json",
                "linear_group": group,
                "prefill_ms": lat["prefill_ms"],
                "decode_ms": lat["decode_ms"],
                "conversion_ms": lat["conversion_ms"],
                "latency_cost": lat["total_ms"],
                "decode_backend": lat["decode_backend"],
                "output_tokens": output_tokens,
                "supported": "True",
                "latency_reason": "",
            }
        )
        candidates.append(item)

    dense_latency = {row["module_name"]: f(row, "latency_cost") for row in candidates if row["method"] == "dense_bf16"}
    dense_quality = {row["module_name"]: f(row, "quality_cost") for row in candidates if row["method"] == "dense_bf16"}
    for row in candidates:
        base_lat = dense_latency.get(row["module_name"], f(row, "latency_cost"))
        base_q = dense_quality.get(row["module_name"], 0.0)
        row["latency_gain_vs_dense"] = base_lat - f(row, "latency_cost")
        row["quality_delta_vs_dense"] = f(row, "quality_cost") - base_q

    out_path = args.output_root / "costs" / "module_method_candidates.csv"
    write_csv(out_path, candidates)
    write_csv(args.output_root / "costs" / "missing_latency.csv", missing_latency)
    write_json(
        args.output_root / "costs" / "build_cost_table_metadata.json",
        {
            "latency_source": "policy_json",
            "policy_json_path": str(args.policy_json),
            "quality_formula": args.quality_formula,
            "candidate_rows": len(candidates),
            "missing_latency_rows": len(missing_latency),
            "methods": list(METHODS),
            "scenario": "normal_01",
        },
    )
    print(f"wrote {len(candidates)} candidate rows to {out_path}")
    if missing_latency:
        print(f"missing latency rows: {len(missing_latency)}")


def load_latency_from_policy_json(policy_path: Path) -> dict[str, dict[str, Any]]:
    """Re-export from common_pareto for local use."""
    from common_pareto import load_latency_from_policy_json as _load

    return _load(policy_path)


if __name__ == "__main__":
    main()
