#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from common_quality import DEBUG_ROOT, parse_policy, write_csv, write_json, write_run_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Llama2 quality modeling experiment.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors_path = args.output_root / "sensitivity" / "module_method_errors.csv"
    ablations_path = args.output_root / "ablations" / "policy_quality_results.csv"
    summary_dir = args.output_root / "summary"
    error_rows = read_csv(errors_path) if errors_path.exists() else []
    ablation_rows = read_csv(ablations_path) if ablations_path.exists() else []

    module_summary = summarize_module_errors(error_rows)
    policy_summary = summarize_policies(ablation_rows)
    proxy_scores = summarize_policy_proxy(error_rows, policy_summary)
    proxy_correlations = summarize_proxy_correlations(proxy_scores)
    formula = recommended_formula(error_rows, ablation_rows)

    write_csv(summary_dir / "module_quality_features.csv", error_rows)
    write_csv(summary_dir / "module_error_summary.csv", module_summary)
    write_csv(summary_dir / "policy_quality_eval.csv", policy_summary)
    write_csv(summary_dir / "policy_proxy_scores.csv", proxy_scores)
    write_csv(summary_dir / "proxy_correlation.csv", proxy_correlations)
    write_json(summary_dir / "recommended_proxy_formula.json", formula)
    write_analysis(summary_dir / "analysis.md", module_summary, policy_summary, proxy_correlations, formula)
    write_run_metadata(
        summary_dir / "summary_metadata.json",
        {
            "module_error_rows": len(error_rows),
            "policy_rows": len(ablation_rows),
            "outputs": [
                "module_quality_features.csv",
                "module_error_summary.csv",
                "policy_quality_eval.csv",
                "policy_proxy_scores.csv",
                "proxy_correlation.csv",
                "recommended_proxy_formula.json",
                "analysis.md",
            ],
        },
    )
    print(f"summary written to {summary_dir}")


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except (TypeError, ValueError):
        return default


def summarize_module_errors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("method", ""), row.get("module_family", ""), row.get("layer_bucket", ""))
        groups.setdefault(key, []).append(row)
    out = []
    for (method, family, bucket), items in sorted(groups.items()):
        values = [f(row, "local_rel_mse") for row in items]
        out.append(
            {
                "method": method,
                "module_family": family,
                "layer_bucket": bucket,
                "modules": len(items),
                "local_rel_mse_mean": mean(values) if values else 0.0,
                "local_rel_mse_max": max(values) if values else 0.0,
                "local_rmse_over_rms_mean": mean(math.sqrt(max(v, 0.0)) for v in values) if values else 0.0,
            }
        )
    return out


def summarize_policies(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dense_nll = next((f(row, "nll") for row in rows if row.get("method") == "dense_bf16"), None)
    dense_acc = next((f(row, "arc_acc") for row in rows if row.get("method") == "dense_bf16"), None)
    out = []
    for row in rows:
        item = dict(row)
        if dense_nll is not None:
            item["nll_delta_recomputed"] = f(row, "nll") - dense_nll
        if dense_acc is not None and row.get("arc_acc", "") != "":
            item["arc_acc_delta_vs_dense"] = f(row, "arc_acc") - dense_acc
        out.append(item)
    return out


def summarize_policy_proxy(error_rows: list[dict[str, Any]], policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    module_names = sorted({row.get("module_name", "") for row in error_rows if row.get("module_name", "")})
    error_by_method_module = {
        (row.get("method", ""), row.get("module_name", "")): row for row in error_rows
    }
    out = []
    for row in policy_rows:
        method = row.get("method", "")
        policy = row.get("policy", "")
        selected = parse_policy(policy, module_names) if method != "dense_bf16" else set()
        score = 0.0
        local_rel_mse_sum = 0.0
        numel_sum = 0.0
        for name in selected:
            erow = error_by_method_module.get((method, name))
            if erow is None:
                continue
            local_rel_mse = f(erow, "local_rel_mse")
            numel = f(erow, "numel")
            local_rel_mse_sum += local_rel_mse
            numel_sum += numel
            score += local_rel_mse * math.log1p(max(numel, 0.0)) * layer_weight(erow) * family_weight(erow)
        item = dict(row)
        item.update(
            {
                "proxy_score": score,
                "proxy_score_per_module": score / len(selected) if selected else 0.0,
                "selected_proxy_modules": len(selected),
                "local_rel_mse_sum": local_rel_mse_sum,
                "selected_numel_sum": numel_sum,
            }
        )
        out.append(item)
    return out


def layer_weight(row: dict[str, Any]) -> float:
    bucket = row.get("layer_bucket", "")
    return {
        "layers_00_07": 1.10,
        "layers_08_15": 1.00,
        "layers_16_23": 1.00,
        "layers_24_31": 1.15,
    }.get(bucket, 1.0)


def family_weight(row: dict[str, Any]) -> float:
    return {
        "attention": 1.0,
        "mlp": 1.0,
    }.get(row.get("module_family", ""), 1.0)


def summarize_proxy_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for method in sorted({row.get("method", "") for row in rows if row.get("method") != "dense_bf16"}):
        items = [row for row in rows if row.get("method") == method]
        for metric in ("nll_delta_recomputed", "arc_acc_delta_vs_dense", "arc_acc_norm"):
            pairs = [(f(row, "proxy_score"), f(row, metric)) for row in items if row.get(metric, "") != ""]
            if len(pairs) < 3:
                continue
            xs = [x for x, _ in pairs]
            ys = [y for _, y in pairs]
            out.append(
                {
                    "method": method,
                    "metric": metric,
                    "rows": len(pairs),
                    "pearson": pearson(xs, ys),
                    "spearman": spearman(xs, ys),
                }
            )
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (den_x * den_y) if den_x and den_y else 0.0


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            out[indexed[k][0]] = rank
        i = j
    return out


def recommended_formula(error_rows: list[dict[str, Any]], policy_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": "quality_proxy_v0",
        "formula": "quality_cost = local_rel_mse * log1p(numel) * layer_weight * module_family_weight",
        "inputs": ["local_rel_mse", "numel", "layer", "module_family"],
        "defaults": {
            "layer_weight": {
                "layers_00_07": 1.10,
                "layers_08_15": 1.00,
                "layers_16_23": 1.00,
                "layers_24_31": 1.15,
                "other": 1.00,
            },
            "module_family_weight": {
                "attention": 1.00,
                "mlp": 1.00,
                "other": 1.00,
            },
        },
        "calibration_status": "first_pass_policy_correlation_available",
        "module_error_rows": len(error_rows),
        "policy_rows": len(policy_rows),
    }


def write_analysis(
    path: Path,
    module_summary: list[dict[str, Any]],
    policy_summary: list[dict[str, Any]],
    proxy_correlations: list[dict[str, Any]],
    formula: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Llama2 Quality Modeling Analysis",
        "",
        "## Outputs",
        "",
        "- `module_quality_features.csv`: per-module, per-method local error and statistics.",
        "- `module_error_summary.csv`: grouped local error summary.",
        "- `policy_quality_eval.csv`: mixed-policy NLL and optional arc_easy results.",
        "- `policy_proxy_scores.csv`: mixed-policy proxy score joined with quality metrics.",
        "- `proxy_correlation.csv`: Pearson/Spearman correlation between proxy score and quality metrics.",
        "- `recommended_proxy_formula.json`: first-pass proxy formula.",
        "",
        "## Current Proxy",
        "",
        f"`{formula['formula']}`",
        "",
        "The proxy is a first-pass fit target; current correlations use the collected mixed-policy rows.",
        "",
        "## Highest Local Error Groups",
        "",
    ]
    ranked = sorted(module_summary, key=lambda row: float(row.get("local_rel_mse_mean", 0.0)), reverse=True)[:10]
    for row in ranked:
        lines.append(
            f"- {row.get('method')} {row.get('module_family')} {row.get('layer_bucket')}: "
            f"mean local_rel_mse={row.get('local_rel_mse_mean')}"
        )
    lines.extend(["", "## Proxy Correlation", ""])
    for row in proxy_correlations:
        lines.append(
            f"- {row.get('method')} vs {row.get('metric')}: "
            f"pearson={row.get('pearson')}, spearman={row.get('spearman')}, rows={row.get('rows')}"
        )
    lines.extend(["", "## Policy Rows", "", f"Collected policy rows: {len(policy_summary)}", ""])
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
