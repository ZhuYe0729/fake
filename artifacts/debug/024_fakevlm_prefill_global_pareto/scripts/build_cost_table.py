#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common_fakevlm_pareto import DEBUG_ROOT, METHODS, f, parse_batches, read_csv, read_json, source_021_latency, write_csv, write_json


QUALITY_VARIANT = "final_layer_type"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FakeVLM per-module method quality/latency candidate tables.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--batches", default="all")
    parser.add_argument("--latency-source", choices=["latency_model", "manual_profile"], default="latency_model")
    parser.add_argument("--metric", default="output_rel_mse")
    parser.add_argument("--coefficients-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batches = parse_batches(args.batches)
    coef_path = args.coefficients_json or args.output_root / "global_coefficients" / "proxy_ablation_coefficients.json"
    coefficients = read_json(coef_path)
    local_rows = read_csv(args.output_root / "sensitivity" / "module_method_local_errors.csv")
    all_rows = []
    metadata = {"batches": batches, "latency_source": args.latency_source, "coefficients_json": str(coef_path), "rows_by_batch": {}}
    for batch in batches:
        latency = latency_map(source_021_latency(batch, args.latency_source))
        rows = []
        missing = []
        for row in local_rows:
            method = row["method"]
            if method not in METHODS:
                continue
            key = (method, int(f(row, "out_features")), int(f(row, "in_features")))
            lrow = latency.get(key)
            if lrow is None:
                missing.append({"batch_size": batch, "module_name": row["module_name"], "method": method, "n": key[1], "k": key[2]})
                continue
            item = dict(row)
            qcost, qmeta = quality_cost(row, coefficients, args.metric)
            item.update(
                {
                    "batch_size": batch,
                    "quality_formula": "global_coef_final_layer_type",
                    "quality_model_variant": qmeta["variant"],
                    "quality_source_method": qmeta["source_method"],
                    "quality_local_error_metric": args.metric,
                    "quality_cost": qcost,
                    "global_coef": qmeta.get("global_coef", ""),
                    "layer_coef": qmeta.get("layer_coef", ""),
                    "type_coef": qmeta.get("type_coef", ""),
                    "latency_source": args.latency_source,
                    "prefill_ms": f(lrow, "latency_ms"),
                    "latency_cost": f(lrow, "latency_ms"),
                    "supported": lrow.get("supported", "True"),
                    "latency_reason": lrow.get("reason", ""),
                }
            )
            rows.append(item)
        add_dense_deltas(rows)
        write_csv(args.output_root / "costs" / f"batch_{batch}" / "module_method_candidates.csv", rows)
        write_csv(args.output_root / "costs" / f"batch_{batch}" / "missing_latency.csv", missing)
        all_rows.extend(rows)
        metadata["rows_by_batch"][batch] = {"candidate_rows": len(rows), "missing_latency_rows": len(missing)}
    write_csv(args.output_root / "costs" / "module_method_candidates_all_batches.csv", all_rows)
    write_json(args.output_root / "costs" / "build_cost_table_metadata.json", metadata)
    print(f"wrote cost tables for batches={batches}")


def latency_map(rows: list[dict[str, Any]]) -> dict[tuple[str, int, int], dict[str, Any]]:
    out = {}
    for row in rows:
        out[(row["method"], int(f(row, "n")), int(f(row, "k")))] = row
    return out


def quality_cost(row: dict[str, Any], coefficients: dict[str, Any], metric: str) -> tuple[float, dict[str, Any]]:
    method = row["method"]
    if method == "dense_bf16":
        return 0.0, {"variant": "dense_zero", "source_method": "dense_bf16"}
    fit = coefficients[method][QUALITY_VARIANT]
    layer = str(int(f(row, "layer")))
    typ = row["module_type"]
    global_coef = float(fit["global_coef"])
    layer_coef = float(fit["layer_coef"][layer])
    type_coef = float(fit["type_coef"][typ])
    cost = f(row, metric) * global_coef * layer_coef * type_coef
    return cost, {
        "variant": fit["variant"],
        "source_method": method,
        "global_coef": global_coef,
        "layer_coef": layer_coef,
        "type_coef": type_coef,
    }


def add_dense_deltas(rows: list[dict[str, Any]]) -> None:
    dense_latency = {row["module_name"]: f(row, "latency_cost") for row in rows if row["method"] == "dense_bf16"}
    for row in rows:
        base = dense_latency.get(row["module_name"], f(row, "latency_cost"))
        row["latency_gain_vs_dense"] = base - f(row, "latency_cost")
        row["quality_delta_vs_dense"] = f(row, "quality_cost")


if __name__ == "__main__":
    main()
