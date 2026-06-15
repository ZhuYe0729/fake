#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import mean
from typing import Any

from build_cost_table import FORMULAS
from common_pareto import DEBUG_ROOT, QUALITY_ROOT, f, load_module_quality_rows, quality_cost, read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare quality proxy formulas on existing policy ablations.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--quality-root", type=Path, default=QUALITY_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    error_rows = [row for row in load_module_quality_rows(args.quality_root) if row.get("method") != "marlin_nvfp4"]
    module_names = sorted({row["module_name"] for row in error_rows if row.get("method") == "dense_bf16"})
    error_by_method_module = {(row["method"], row["module_name"]): row for row in error_rows}
    eval_sets = [
        ("arc_easy_limit128", args.quality_root / "ablations" / "policy_quality_results.csv"),
        ("arc_challenge_limit128", args.quality_root / "arc_challenge_limit128" / "ablations" / "policy_quality_results.csv"),
    ]
    rows = []
    for eval_name, path in eval_sets:
        if not path.exists():
            continue
        policies = add_deltas(read_csv(path))
        for formula in FORMULAS:
            scored = []
            for row in policies:
                method = row.get("method", "")
                if method == "dense_bf16":
                    score = 0.0
                else:
                    selected = parse_policy(row.get("policy", ""), module_names)
                    score = sum(
                        quality_cost(error_by_method_module[(method, name)], formula)
                        for name in selected
                        if (method, name) in error_by_method_module
                    )
                item = dict(row)
                item["proxy_score"] = score
                scored.append(item)
            for method in sorted({row["method"] for row in scored if row["method"] != "dense_bf16"}):
                items = [row for row in scored if row["method"] == method]
                for metric in ("nll_delta_recomputed", "arc_acc_delta_vs_dense", "arc_acc_norm"):
                    pairs = [(f(row, "proxy_score"), f(row, metric)) for row in items if row.get(metric, "") != ""]
                    if len(pairs) < 3:
                        continue
                    xs = [x for x, _ in pairs]
                    ys = [y for _, y in pairs]
                    rows.append(
                        {
                            "eval_set": eval_name,
                            "formula": formula,
                            "method": method,
                            "metric": metric,
                            "rows": len(pairs),
                            "pearson": pearson(xs, ys),
                            "spearman": spearman(xs, ys),
                        }
                    )
    write_csv(args.output_root / "quality" / "formula_correlation.csv", rows)
    write_json(
        args.output_root / "quality" / "formula_analysis_metadata.json",
        {
            "formulas": list(FORMULAS),
            "rows": len(rows),
            "eval_sets": [name for name, path in eval_sets if path.exists()],
        },
    )
    print(f"wrote {len(rows)} formula correlation rows")


def add_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def parse_policy(policy: str, module_names: list[str]) -> set[str]:
    if policy == "none":
        return set()
    if policy == "all":
        return set(module_names)
    if policy.startswith("family:"):
        family = policy.split(":", 1)[1]
        return {name for name in module_names if module_family(name) == family}
    if policy.startswith("type:"):
        typ = policy.split(":", 1)[1]
        return {name for name in module_names if name.rsplit(".", 1)[-1] == typ}
    if policy.startswith("bucket:"):
        bucket = policy.split(":", 1)[1]
        return {name for name in module_names if layer_bucket(layer_index(name)) == bucket}
    if policy.startswith("layer:"):
        layer = int(policy.split(":", 1)[1])
        return {name for name in module_names if layer_index(name) == layer}
    if policy.startswith("layer_family:"):
        spec = policy.split(":", 1)[1]
        layer_text, family = spec.split(":", 1)
        layer = int(layer_text)
        return {name for name in module_names if layer_index(name) == layer and module_family(name) == family}
    if policy.startswith("module:"):
        return {policy.split(":", 1)[1]}
    raise ValueError(f"unsupported policy: {policy}")


def layer_index(module_name: str) -> int:
    parts = module_name.split(".")
    return int(parts[2]) if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers" else -1


def layer_bucket(layer: int) -> str:
    if 0 <= layer <= 7:
        return "layers_00_07"
    if 8 <= layer <= 15:
        return "layers_08_15"
    if 16 <= layer <= 23:
        return "layers_16_23"
    if 24 <= layer <= 31:
        return "layers_24_31"
    return "other"


def module_family(module_name: str) -> str:
    typ = module_name.rsplit(".", 1)[-1]
    if typ in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return "attention"
    if typ in {"gate_proj", "up_proj", "down_proj"}:
        return "mlp"
    return "other"


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


if __name__ == "__main__":
    main()
