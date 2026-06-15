#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common_pareto import DEBUG_ROOT, FAKE_ROOT, METHODS, ORACLE_SUMMARY_ROOT, f, normalize_group_name, read_csv, read_json, write_csv, write_json


SOURCE_014_ROOT = FAKE_ROOT / "artifacts/debug/014_llama2_prefill_loss_modeling"
SOURCE_015_ROOT = FAKE_ROOT / "artifacts/debug/015_llama2_prefill_kernel_loss_modeling"
PREPARED_ROOT = FAKE_ROOT / "artifacts/results/main/003_llama2_7b_arc_easy_accuracy/prepared"
DEFAULT_CANDIDATE_METHODS = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4")
QUALITY_VARIANT = "final_layer_type"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-module method quality/latency candidate table.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--latency-source", choices=["existing", "fresh"], default="existing")
    parser.add_argument("--candidate-methods", default=",".join(DEFAULT_CANDIDATE_METHODS))
    parser.add_argument("--local-error-metric", default="output_rel_mse")
    parser.add_argument("--coefficients-json", type=Path, default=DEBUG_ROOT / "global_coefficients" / "proxy_ablation_coefficients.json")
    parser.add_argument("--marlin-local-errors", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.candidate_methods)
    coefficients = load_coefficients(args.coefficients_json)
    quality_rows = load_quality_rows(methods, args.local_error_metric, args.marlin_local_errors, coefficients)
    latency_rows = load_latency_rows(args.output_root, args.latency_source)
    latency = latency_map(latency_rows)
    candidates: list[dict[str, Any]] = []
    missing_latency: list[dict[str, Any]] = []
    for qrow in quality_rows:
        method = qrow.get("method", "")
        if method not in methods:
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
        qcost, qmeta = module_quality_cost(qrow, coefficients, args.local_error_metric)
        item.update(
            {
                "quality_formula": "global_coef_final_layer_type",
                "quality_model_variant": qmeta["variant"],
                "quality_source_method": qmeta["source_method"],
                "quality_local_error_metric": args.local_error_metric,
                "quality_cost": qcost,
                "global_coef": qmeta.get("global_coef", ""),
                "layer_coef": qmeta.get("layer_coef", ""),
                "type_coef": qmeta.get("type_coef", ""),
                "prepared_artifact": str(PREPARED_ROOT / method / "model.pt") if method != "dense_bf16" else "",
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
            "quality_formula": "global_coef_final_layer_type",
            "coefficients_json": str(args.coefficients_json),
            "local_error_metric": args.local_error_metric,
            "candidate_rows": len(candidates),
            "missing_latency_rows": len(missing_latency),
            "methods": list(methods),
        },
    )
    print(f"wrote {len(candidates)} candidate rows to {out_path}")
    if missing_latency:
        print(f"missing latency rows: {len(missing_latency)}")


def parse_methods(spec: str) -> tuple[str, ...]:
    methods = tuple(item.strip() for item in spec.split(",") if item.strip())
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; supported={METHODS}")
    if "dense_bf16" not in methods:
        raise ValueError("dense_bf16 must be included as the zero-quality baseline")
    return methods


def load_coefficients(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    out: dict[str, dict[str, Any]] = {}
    for method, variants in payload.items():
        fit = variants.get(QUALITY_VARIANT)
        if fit is not None:
            out[method] = fit
    return out


def load_quality_rows(
    methods: tuple[str, ...],
    metric: str,
    marlin_local_errors: Path | None,
    coefficients: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_method: dict[str, list[dict[str, Any]]] = {}
    kernel_rows = read_csv(SOURCE_015_ROOT / "sensitivity" / "module_method_kernel_local_errors.csv")
    local_rows = read_csv(SOURCE_014_ROOT / "sensitivity" / "module_method_local_errors.csv")
    rows_by_method["dense_nvfp4"] = [row for row in kernel_rows if row.get("method") == "dense_nvfp4"]
    rows_by_method["sparse_nvfp4"] = [row for row in kernel_rows if row.get("method") == "sparse_nvfp4"]
    rows_by_method["sparse_bf16"] = [row for row in local_rows if row.get("method") == "sparse_bf16"]
    if "marlin_nvfp4" in methods:
        if marlin_local_errors is None:
            raise RuntimeError(
                "marlin_nvfp4 was requested as a Pareto candidate, but no --marlin-local-errors file was provided. "
                "Do not alias Marlin quality to dense_nvfp4."
            )
        rows_by_method["marlin_nvfp4"] = [row for row in read_csv(marlin_local_errors) if row.get("method") == "marlin_nvfp4"]
    dense_source = rows_by_method.get("dense_nvfp4") or [row for row in local_rows if row.get("method") == "dense_nvfp4"]
    rows_by_method["dense_bf16"] = [dense_row(row) for row in dense_source]

    out: list[dict[str, Any]] = []
    for method in methods:
        if method != "dense_bf16" and method not in coefficients:
            raise RuntimeError(f"missing {QUALITY_VARIANT} coefficients for {method}")
        rows = rows_by_method.get(method, [])
        if not rows:
            raise RuntimeError(f"no local-error rows for {method}")
        missing_metric = [row["module_name"] for row in rows if method != "dense_bf16" and row.get(metric, "") == ""]
        if missing_metric:
            raise RuntimeError(f"{method} rows missing metric {metric}: sample={missing_metric[:3]}")
        out.extend(rows)
    return sorted(out, key=lambda row: (int(f(row, "module_index")), METHODS.index(row["method"])))


def dense_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["method"] = "dense_bf16"
    for key in ("weight_mse", "weight_rel_mse", "weight_rmse_over_rms", "weight_max_abs_error", "output_mse", "output_rel_mse", "output_rmse_over_rms", "output_max_abs_error"):
        out[key] = 0.0
    return out


def module_quality_cost(row: dict[str, Any], coefficients: dict[str, dict[str, Any]], metric: str) -> tuple[float, dict[str, Any]]:
    method = row["method"]
    if method == "dense_bf16":
        return 0.0, {"variant": "dense_zero", "source_method": "dense_bf16"}
    fit = coefficients[method]
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
