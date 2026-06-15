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
    get_latency_for_module,
    load_latency_from_pred_candidates,
    load_module_quality_rows,
    normalize_group_name,
    quality_cost,
    read_csv,
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
    parser = argparse.ArgumentParser(description="Build per-module method quality/latency candidate table for normal_02.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--quality-formula", choices=FORMULAS, default="local_rel_mse_log_numel_layer_family")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quality_rows = load_module_quality_rows()
    latency_lookup = load_latency_from_pred_candidates()
    output_tokens = SCENARIO["output_tokens"]

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
                "latency_source": "003_pred_candidates_normal_02",
                "linear_group": group,
                "prefill_ms": lat["prefill_ms"],
                "decode_ms": lat["decode_ms"],
                "conversion_ms": lat["conversion_ms"],
                "latency_cost": lat["total_ms"],
                "prefill_backend": lat["prefill_backend"],
                "decode_backend": lat["decode_backend"],
                "output_tokens": output_tokens,
                "supported": str(lat["supported"]),
                "unsupported_reason": lat["reason"] if not lat["supported"] else "",
            }
        )
        candidates.append(item)

    # Verify total_ms formula
    total_ms_ok = True
    total_ms_errors: list[dict[str, Any]] = []
    for row in candidates:
        expected_total = f(row, "prefill_ms") + output_tokens * f(row, "decode_ms") + f(row, "conversion_ms")
        actual_total = f(row, "latency_cost")
        if abs(expected_total - actual_total) > 0.01:
            total_ms_ok = False
            total_ms_errors.append(
                {
                    "module_name": row["module_name"],
                    "method": row["method"],
                    "expected_total_ms": expected_total,
                    "actual_total_ms": actual_total,
                    "diff": abs(expected_total - actual_total),
                }
            )

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
            "latency_source": "003_pred_candidates_normal_02",
            "pred_candidates_path": str(load_latency_from_pred_candidates.__defaults__[0]),
            "quality_formula": args.quality_formula,
            "candidate_rows": len(candidates),
            "missing_latency_rows": len(missing_latency),
            "methods": list(METHODS),
            "scenario": SCENARIO,
            "total_ms_formula_verified": "prefill_ms + output_tokens * decode_ms + conversion_ms",
            "total_ms_formula_ok": total_ms_ok,
            "total_ms_formula_errors": len(total_ms_errors),
        },
    )
    print(f"wrote {len(candidates)} candidate rows to {out_path}")
    print(f"total_ms formula OK: {total_ms_ok}")
    if total_ms_errors:
        for err in total_ms_errors[:5]:
            print(f"  total_ms mismatch: {err}")
    if missing_latency:
        print(f"missing latency rows: {len(missing_latency)}")


if __name__ == "__main__":
    main()
