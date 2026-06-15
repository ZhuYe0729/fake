#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common_pareto import DEBUG_ROOT, ORACLE_SUMMARY_ROOT, QUALITY_ROOT, f, read_csv, write_csv


UNIFORM_METHODS = [
    ("dense_bf16", "all", True),
    ("dense_nvfp4", "all", True),
    ("sparse_bf16", "all", True),
    ("sparse_nvfp4", "all", True),
    ("marlin_nvfp4", "all", False),
]

SINGLE_QUALITY_CSV = QUALITY_ROOT / "arc_challenge_limit128" / "ablations" / "policy_quality_results.csv"
METHOD_COST_CSV = DEBUG_ROOT / "summary" / "method_cost_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified Pareto + uniform baseline comparison table.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--include-marlin", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []

    e2e_csv = args.output_root / "validation" / "pareto_validation_joined.csv"
    if e2e_csv.exists():
        pareto_rows = read_csv(e2e_csv)
        for row in pareto_rows:
            pi = int(f(row, "point_index"))
            e2e_mean = f(row, "e2e_prefill_mean_ms")
            nll_val = float(row.get("nll", 0) or 0)
            dense_nll = nll_for_point(pareto_rows, 0)
            dense_e2e = e2e_for_point(pareto_rows, 0)
            rows.append(
                {
                    "row_type": "pareto",
                    "label": f"point_{pi:03d}",
                    "point_index": pi,
                    "quality_cost": f(row, "quality_cost"),
                    "predicted_linear_latency_ms": f(row, "predicted_linear_latency_ms"),
                    "e2e_prefill_mean_ms": e2e_mean,
                    "e2e_speedup_vs_dense": dense_e2e / e2e_mean if dense_e2e and e2e_mean else "",
                    "nll": nll_val,
                    "nll_delta_vs_dense": nll_val - dense_nll if dense_nll else "",
                    "arc_acc": row.get("arc_acc", ""),
                    "arc_acc_norm": row.get("arc_acc_norm", ""),
                    "backend_counts": row.get("backend_counts", ""),
                    "source": "pareto_validation",
                }
            )

    uniform_quality = load_uniform_quality()
    method_costs = load_method_costs()

    for method, policy, required in UNIFORM_METHODS:
        if not required and not args.include_marlin:
            continue
        e2e_row = load_uniform_e2e(method)
        qkey = "dense_nvfp4" if method == "marlin_nvfp4" else method
        q = uniform_quality.get(qkey, {})
        mc = method_costs.get(method, {})
        e2e_mean = float(e2e_row.get("e2e_prefill_mean_ms", e2e_row.get("e2e_ms", 0)) or 0) if e2e_row else 0.0
        dense_e2e = uniform_e2e_value("dense_bf16")
        dense_nll = float(q.get("nll", 0) or 0) if method == "dense_bf16" else uniform_quality.get("dense_bf16", {}).get("nll", 0)
        nll_val = float(q.get("nll", 0) or 0)
        source_tag = "003_oracle_summary"
        if not e2e_row:
            source_tag = "missing_e2e"
            e2e_mean = 0.0

        rows.append(
            {
                "row_type": "uniform",
                "label": f"all_{method}",
                "point_index": "",
                "quality_cost": mc.get("quality_sum", ""),
                "predicted_linear_latency_ms": mc.get("latency_sum_ms", ""),
                "e2e_prefill_mean_ms": e2e_mean if e2e_mean else "",
                "e2e_speedup_vs_dense": dense_e2e / e2e_mean if dense_e2e and e2e_mean else "",
                "nll": nll_val if nll_val else "",
                "nll_delta_vs_dense": (float(nll_val) - float(dense_nll)) if nll_val and dense_nll else "",
                "arc_acc": q.get("arc_acc", ""),
                "arc_acc_norm": q.get("arc_acc_norm", ""),
                "backend_counts": f"{{{method}: 224}}" if method != "dense_bf16" else "{dense_bf16: 224}",
                "source": source_tag,
            }
        )

    out = args.output_root / "summary" / "prefill_only_comparison.csv"
    write_csv(out, rows)
    print(f"wrote {len(rows)} rows to {out}")


def load_uniform_quality() -> dict[str, dict]:
    if not SINGLE_QUALITY_CSV.exists():
        return {}
    mapping: dict[str, dict] = {}
    for row in read_csv(SINGLE_QUALITY_CSV):
        method = row.get("method", "")
        policy = row.get("policy", "")
        if method in ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4"):
            if method == "dense_bf16" and policy == "none":
                mapping[method] = row
            elif method != "dense_bf16" and policy == "all":
                mapping[method] = row
    return mapping


def load_method_costs() -> dict[str, dict]:
    if not METHOD_COST_CSV.exists():
        return {}
    mapping: dict[str, dict] = {}
    for row in read_csv(METHOD_COST_CSV):
        mapping[row["method"]] = row
    return mapping


def load_uniform_e2e(method: str) -> dict | None:
    path = ORACLE_SUMMARY_ROOT / "single" / method / "prefill_only" / "llama2-7b_full_e2e.csv"
    if not path.exists():
        return None
    rows = read_csv(path)
    if not rows:
        return None
    row = rows[0]
    return {
        "e2e_prefill_mean_ms": row.get("prefill_ms", row.get("e2e_ms", "0")),
        "replaced_linear_count": row.get("replaced_linear_count", ""),
        "skipped_linear_count": row.get("skipped_linear_count", ""),
        "backend_counts": row.get("backend_counts", ""),
    }


def uniform_e2e_value(method: str) -> float:
    row = load_uniform_e2e(method)
    if not row:
        return 0.0
    return float(row.get("e2e_prefill_mean_ms", 0) or 0)


def nll_for_point(rows: list[dict], point: int) -> float:
    for row in rows:
        if int(f(row, "point_index")) == point:
            val = row.get("nll", "")
            return float(val) if val else 0.0
    return 0.0


def e2e_for_point(rows: list[dict], point: int) -> float:
    for row in rows:
        if int(f(row, "point_index")) == point:
            return f(row, "e2e_prefill_mean_ms")
    return 0.0


if __name__ == "__main__":
    main()
