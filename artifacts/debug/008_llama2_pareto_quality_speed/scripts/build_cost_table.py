#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common_pareto import (
    DEBUG_ROOT,
    METHODS,
    ORACLE_SUMMARY_ROOT,
    f,
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
    parser = argparse.ArgumentParser(description="Build per-module method quality/latency candidate table.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--latency-source", choices=["existing", "fresh"], default="existing")
    parser.add_argument("--quality-formula", choices=FORMULAS, default="local_rel_mse_log_numel_layer_family")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quality_rows = load_module_quality_rows()
    latency_rows = load_latency_rows(args.output_root, args.latency_source)
    latency = latency_map(latency_rows)
    candidates: list[dict[str, Any]] = []
    missing_latency: list[dict[str, Any]] = []
    for qrow in quality_rows:
        method = qrow.get("method", "")
        if method not in METHODS:
            continue
        module_name = str(qrow["module_name"])
        group = normalize_group_name(module_name)
        key = (method, group, int(f(qrow, "out_features")), int(f(qrow, "in_features")))
        lrow = latency.get(key)
        if lrow is None:
            missing_latency.append({"method": method, "module_name": module_name, "linear_group": group, "n": key[2], "k": key[3]})
            continue
        prefill_ms = f(lrow, "prefill_ms")
        item = dict(qrow)
        item.update(
            {
                "quality_formula": args.quality_formula,
                "quality_cost": quality_cost(qrow, args.quality_formula),
                "latency_source": args.latency_source,
                "linear_group": group,
                "prefill_ms": prefill_ms,
                "latency_cost": prefill_ms,
                "supported": str(lrow.get("supported", "True")),
                "latency_reason": lrow.get("reason", ""),
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
            "latency_source": args.latency_source,
            "quality_formula": args.quality_formula,
            "candidate_rows": len(candidates),
            "missing_latency_rows": len(missing_latency),
            "methods": list(METHODS),
        },
    )
    print(f"wrote {len(candidates)} candidate rows to {out_path}")
    if missing_latency:
        print(f"missing latency rows: {len(missing_latency)}")


def load_latency_rows(output_root: Path, source: str) -> list[dict[str, Any]]:
    if source == "fresh":
        path = output_root / "latency" / "prefill_latency.csv"
        if not path.exists():
            raise FileNotFoundError(f"fresh latency file missing: {path}")
        return read_csv(path)
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        path = ORACLE_SUMMARY_ROOT / "single" / method / "prefill_only" / "llama2-7b_linear_summary.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        for row in read_csv(path):
            if row.get("linear_group") == "__TOTAL__":
                continue
            row = dict(row)
            row["method"] = method
            row["source"] = "existing_oracle_summary"
            rows.append(row)
    return rows


def latency_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    out = {}
    for row in rows:
        if row.get("linear_group") == "__TOTAL__":
            continue
        method = str(row.get("method") or row.get("candidate"))
        group = str(row["linear_group"])
        n = int(f(row, "n"))
        k = int(f(row, "k"))
        supported = str(row.get("supported", "True")).lower() in {"true", "1", "yes"}
        if not supported:
            continue
        out[(method, group, n, k)] = row
    return out


if __name__ == "__main__":
    main()
