#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
PARETO_ROOT = ROOT / "artifacts" / "debug" / "030_mirror_global_pareto"
OUT_ROOT = ROOT / "artifacts" / "debug" / "031_mirror_sparse_bf16_additivity_debug"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MIRROR sparse_bf16 additive quality-model failures.")
    parser.add_argument("--pareto-root", type=Path, default=PARETO_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    tables_dir = args.output_root / "tables"
    plots_dir = args.output_root / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    costs = load_costs(args.pareto_root / "costs_keyfix_genimage" / "batch_16" / "module_method_candidates.csv")
    policies = load_policy_catalog(args.pareto_root)
    measurements = []
    measurements.extend(load_stratified_partial(args.pareto_root, policies, costs))
    measurements.extend(load_joined(args.pareto_root / "validation_keyfix_genimage" / "supplemental_joined.csv", "full_supplemental", costs))
    measurements.extend(load_joined(args.pareto_root / "validation_keyfix_genimage" / "theoretical_joined.csv", "full_theoretical", costs))
    measurements.extend(load_uniform_full(args.pareto_root, costs))
    measurements.extend(load_controlled_partial(args.output_root, costs))
    enrich_baselines(measurements)

    write_csv(tables_dir / "sparse_bf16_policy_measurements.csv", measurements)
    diagnostics = build_diagnostics(measurements)
    write_csv(tables_dir / "sparse_bf16_residual_diagnostics.csv", diagnostics)
    type_rows, bucket_rows = build_type_bucket_tables(measurements)
    count_variance_rows = build_count_variance_table(measurements)
    correlation_rows = build_correlation_table(measurements)
    write_csv(tables_dir / "sparse_bf16_type_summary.csv", type_rows)
    write_csv(tables_dir / "sparse_bf16_layer_bucket_summary.csv", bucket_rows)
    write_csv(tables_dir / "sparse_bf16_same_count_variance.csv", count_variance_rows)
    write_csv(tables_dir / "sparse_bf16_correlation_summary.csv", correlation_rows)

    plot_ratio_vs_nll(measurements, plots_dir)
    plot_predicted_vs_true(measurements, plots_dir)
    plot_residual_vs_ratio(measurements, plots_dir)
    plot_full_policy_comparison(measurements, plots_dir)
    plot_same_count_variance(measurements, plots_dir)
    write_summary(args.output_root / "summary.md", measurements, diagnostics, type_rows, bucket_rows, count_variance_rows, correlation_rows)
    print(f"wrote sparse_bf16 additivity analysis to {args.output_root}")


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except (TypeError, ValueError):
        return default


def parse_counts(value: Any) -> Counter[str]:
    if isinstance(value, dict):
        return Counter({str(k): int(v) for k, v in value.items()})
    if not value:
        return Counter()
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return Counter()
    if not isinstance(parsed, dict):
        return Counter()
    return Counter({str(k): int(v) for k, v in parsed.items()})


def load_costs(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in read_csv(path):
        out[row["module_name"]][row["method"]] = row
    return dict(out)


def load_policy_catalog(root: Path) -> dict[int, dict[str, Any]]:
    out = {}
    for row in read_csv(root / "stratified_keyfix_genimage" / "quality_policies.csv"):
        out[int(f(row, "policy_index"))] = row
    return out


def load_policy_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return root / path


def policy_stats(policy_json: Path, costs: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    policy = load_policy_json(policy_json)
    modules = policy.get("modules", [])
    counts = Counter()
    type_counts = Counter()
    layer_bucket_counts = Counter()
    selected_q = 0.0
    selected_lat = 0.0
    selected_local_error = 0.0
    sparse_q = 0.0
    sparse_local_error = 0.0
    method_switches = 0
    for module in modules:
        name = module.get("module_name") or module.get("name")
        method = module.get("selected_method") or module.get("backend")
        row = costs[name][method]
        counts[method] += 1
        typ = module.get("module_type") or row.get("module_type", "")
        layer = int(float(module.get("layer", row.get("layer", -1))))
        bucket = layer_bucket(layer)
        type_counts[f"{typ}:{method}"] += 1
        layer_bucket_counts[f"{bucket}:{method}"] += 1
        q = float(row["quality_cost"])
        lat = float(row["latency_cost"])
        err = float(row.get("output_rel_mse", 0.0) or 0.0)
        selected_q += q
        selected_lat += lat
        selected_local_error += err
        if method == "sparse_bf16":
            sparse_q += q
            sparse_local_error += err
        if method != "sparse_bf16":
            method_switches += 1
    total = max(len(modules), 1)
    return {
        "count_dense_default": counts.get("dense_default", 0),
        "count_dense_bf16": counts.get("dense_bf16", 0),
        "count_dense_nvfp4": counts.get("dense_nvfp4", 0),
        "count_sparse_bf16": counts.get("sparse_bf16", 0),
        "count_sparse_nvfp4": counts.get("sparse_nvfp4", 0),
        "sparse_ratio": counts.get("sparse_bf16", 0) / total,
        "predicted_quality_cost": selected_q,
        "predicted_linear_latency_ms": selected_lat,
        "sum_selected_local_rel_mse": selected_local_error,
        "sum_sparse_bf16_local_rel_mse": sparse_local_error,
        "sum_sparse_bf16_quality_cost": sparse_q,
        "method_diversity": sum(1 for _method, count in counts.items() if count > 0),
        "non_sparse_count": method_switches,
        "type_counts": dict(type_counts),
        "layer_bucket_counts": dict(layer_bucket_counts),
    }


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


def aggregate_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(f(row, "num_samples")) for row in rows)
    item = {
        "num_samples": total,
        "ce_nll": sum(f(row, "ce_nll") * int(f(row, "num_samples")) for row in rows) / max(total, 1),
    }
    for metric in ("acc", "real_acc", "fake_acc", "bal_acc", "auc", "ap"):
        vals = [f(row, metric) for row in rows if row.get(metric, "") != ""]
        item[metric] = sum(vals) / max(len(vals), 1)
    return item


def load_stratified_partial(root: Path, policies: dict[int, dict[str, Any]], costs: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = read_csv(root / "quality" / "stratified_keyfix_genimage_quality.csv")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(f(row, "policy_index"))].append(row)
    out = []
    for idx, group in grouped.items():
        policy = policies.get(idx)
        if not policy:
            continue
        label = policy.get("label", "")
        counts = parse_counts(policy.get("backend_counts"))
        if "sparse_bf16" not in counts and "sparse_bf16" not in label:
            continue
        policy_path = resolve_path(root, policy["policy_json"])
        stats = policy_stats(policy_path, costs)
        quality = aggregate_quality(group)
        out.append(build_measurement("genimage_partial", label, idx, "", policy_path, stats, quality, "GenImage-partial"))
    return out


def load_joined(path: Path, source: str, costs: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    root = path.parents[1]
    for row in read_csv(path):
        counts = Counter({method.replace("count_", ""): int(f(row, method)) for method in row if method.startswith("count_")})
        label = row.get("label") or row.get("selection_reason") or row.get("key", "")
        if counts.get("sparse_bf16", 0) == 0 and "sparse_bf16" not in label:
            continue
        policy_path = resolve_path(root, row["policy_json"])
        stats = policy_stats(policy_path, costs)
        quality = {
            "num_samples": int(f(row, "quality_num_samples")),
            "ce_nll": f(row, "quality_ce_nll"),
            "bal_acc": f(row, "quality_bal_acc"),
            "auc": f(row, "quality_auc"),
            "ap": f(row, "quality_ap"),
        }
        out.append(build_measurement(source, label, row.get("point_index", ""), row.get("key", ""), policy_path, stats, quality, "ALL-full"))
    return out


def load_uniform_full(root: Path, costs: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    policies = {row["label"]: row for row in read_csv(root / "keyfix_uniform" / "quality_policies.csv")}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_csv(root / "quality" / "keyfix_uniform_quality.csv"):
        grouped[row["label"]].append(row)
    out = []
    for label, group in grouped.items():
        if label not in {"uniform_dense_bf16", "uniform_dense_default", "uniform_sparse_bf16"}:
            continue
        policy = policies[label]
        policy_path = resolve_path(root, policy["policy_json"])
        stats = policy_stats(policy_path, costs)
        quality = aggregate_quality(group)
        out.append(build_measurement("full_uniform", label, policy.get("policy_index", ""), "", policy_path, stats, quality, "ALL-full"))
    return out


def load_controlled_partial(root: Path, costs: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    policy_rows = {int(f(row, "policy_index")): row for row in read_csv(root / "controlled_sparse_bf16" / "quality_policies.csv")}
    quality_rows = read_csv(root / "quality" / "controlled_sparse_bf16_genimage_quality.csv")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in quality_rows:
        grouped[int(f(row, "policy_index"))].append(row)
    out = []
    for idx, group in grouped.items():
        policy = policy_rows.get(idx)
        if not policy:
            continue
        policy_path = resolve_path(root, policy["policy_json"])
        stats = policy_stats(policy_path, costs)
        quality = aggregate_quality(group)
        out.append(build_measurement("controlled_genimage_partial", policy["label"], idx, f"controlled_policy_{idx:03d}", policy_path, stats, quality, "GenImage-controlled-partial"))
    return out


def build_measurement(source: str, label: str, index: Any, key: str, policy_path: Path, stats: dict[str, Any], quality: dict[str, Any], eval_scope: str) -> dict[str, Any]:
    row = {
        "source": source,
        "eval_scope": eval_scope,
        "label": label,
        "policy_index_or_point": index,
        "key": key,
        "policy_json": str(policy_path),
        "true_ce_nll": quality.get("ce_nll", ""),
        "true_bal_acc": quality.get("bal_acc", ""),
        "true_auc": quality.get("auc", ""),
        "true_ap": quality.get("ap", ""),
        "num_samples": quality.get("num_samples", ""),
    }
    for name, value in stats.items():
        if isinstance(value, dict):
            row[name] = json.dumps(value, sort_keys=True)
        else:
            row[name] = value
    return row


def enrich_baselines(rows: list[dict[str, Any]]) -> None:
    by_scope = defaultdict(list)
    for row in rows:
        by_scope[row["eval_scope"]].append(row)
    for scope, group in by_scope.items():
        dense_bf16 = next((row for row in group if row["label"] in {"dense_bf16", "dense_bf16_baseline", "sparse_bf16_lowerr_ratio_0", "sparse_bf16_speed_ratio_0", "uniform_dense_bf16"}), None)
        dense_default = next((row for row in group if row["label"] in {"dense_default", "uniform_dense_default"}), None)
        bf16_nll = f(dense_bf16 or {}, "true_ce_nll")
        dense_nll = f(dense_default or {}, "true_ce_nll")
        for row in group:
            nll = f(row, "true_ce_nll")
            row["true_nll_delta_vs_bf16"] = nll - bf16_nll if bf16_nll else ""
            row["true_nll_delta_vs_dense"] = nll - dense_nll if dense_nll else ""
            row["true_nll_delta_vs_bf16_clipped"] = max(0.0, nll - bf16_nll) if bf16_nll else ""
            row["quality_residual_vs_bf16"] = (max(0.0, nll - bf16_nll) - f(row, "predicted_quality_cost")) if bf16_nll else ""


def build_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("quality_residual_vs_bf16", "") == "":
            continue
        out.append(
            {
                "source": row["source"],
                "label": row["label"],
                "eval_scope": row["eval_scope"],
                "sparse_ratio": row["sparse_ratio"],
                "count_sparse_bf16": row["count_sparse_bf16"],
                "predicted_quality_cost": row["predicted_quality_cost"],
                "true_ce_nll": row["true_ce_nll"],
                "true_nll_delta_vs_bf16_clipped": row["true_nll_delta_vs_bf16_clipped"],
                "quality_residual_vs_bf16": row["quality_residual_vs_bf16"],
                "true_bal_acc": row["true_bal_acc"],
                "sum_sparse_bf16_local_rel_mse": row["sum_sparse_bf16_local_rel_mse"],
                "method_diversity": row["method_diversity"],
                "non_sparse_count": row["non_sparse_count"],
            }
        )
    return sorted(out, key=lambda row: abs(f(row, "quality_residual_vs_bf16")), reverse=True)


def build_type_bucket_tables(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    type_acc: dict[str, list[float]] = defaultdict(list)
    bucket_acc: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        residual = f(row, "quality_residual_vs_bf16", math.nan)
        if math.isnan(residual):
            continue
        for key, value in json.loads(row.get("type_counts", "{}")).items():
            if key.endswith(":sparse_bf16") and value:
                type_acc[key.split(":", 1)[0]].append(residual)
        for key, value in json.loads(row.get("layer_bucket_counts", "{}")).items():
            if key.endswith(":sparse_bf16") and value:
                bucket_acc[key.split(":", 1)[0]].append(residual)
    type_rows = summarize_groups(type_acc)
    bucket_rows = summarize_groups(bucket_acc)
    return type_rows, bucket_rows


def build_count_variance_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    controlled = [row for row in rows if row["source"] == "controlled_genimage_partial"]
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in controlled:
        groups[int(f(row, "count_sparse_bf16"))].append(row)
    out = []
    for count, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        nlls = [f(row, "true_ce_nll") for row in group]
        preds = [f(row, "predicted_quality_cost") for row in group]
        labels = [row["label"] for row in group]
        best = min(group, key=lambda row: f(row, "true_ce_nll"))
        worst = max(group, key=lambda row: f(row, "true_ce_nll"))
        out.append(
            {
                "count_sparse_bf16": count,
                "policies": len(group),
                "nll_min": min(nlls),
                "nll_max": max(nlls),
                "nll_range": max(nlls) - min(nlls),
                "nll_mean": sum(nlls) / len(nlls),
                "predicted_cost_min": min(preds),
                "predicted_cost_max": max(preds),
                "predicted_cost_range": max(preds) - min(preds),
                "best_label": best["label"],
                "worst_label": worst["label"],
                "labels": ",".join(labels),
            }
        )
    return out


def build_correlation_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for source, group in group_by(rows, "source").items():
        usable = [row for row in group if row.get("true_nll_delta_vs_bf16_clipped", "") != ""]
        if len(usable) < 3:
            continue
        y = [f(row, "true_nll_delta_vs_bf16_clipped") for row in usable]
        for feature in ("predicted_quality_cost", "sum_sparse_bf16_local_rel_mse", "sparse_ratio", "count_sparse_bf16", "method_diversity", "non_sparse_count"):
            x = [f(row, feature) for row in usable]
            out.append(
                {
                    "source": source,
                    "feature": feature,
                    "rows": len(usable),
                    "pearson": pearson(x, y),
                    "spearman": pearson(ranks(x), ranks(y)),
                    "rmse_linear_fit": linear_fit_rmse(x, y),
                }
            )
    return sorted(out, key=lambda row: (row["source"], -abs(float(row["spearman"]))))


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def ranks(values: list[float]) -> list[float]:
    ordered = sorted((value, index) for index, value in enumerate(values))
    out = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        rank = (i + j - 1) / 2.0
        for _value, index in ordered[i:j]:
            out[index] = rank
        i = j
    return out


def linear_fit_rmse(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= 0:
        pred = [my] * len(ys)
    else:
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
        intercept = my - slope * mx
        pred = [intercept + slope * x for x in xs]
    return math.sqrt(sum((p - y) ** 2 for p, y in zip(pred, ys)) / len(ys))


def summarize_groups(groups: dict[str, list[float]]) -> list[dict[str, Any]]:
    rows = []
    for name, values in groups.items():
        rows.append(
            {
                "group": name,
                "policies": len(values),
                "mean_residual": sum(values) / len(values),
                "min_residual": min(values),
                "max_residual": max(values),
            }
        )
    return sorted(rows, key=lambda row: abs(float(row["mean_residual"])), reverse=True)


def plot_ratio_vs_nll(rows: list[dict[str, Any]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    for source, group in group_by(rows, "source").items():
        xs = [f(row, "sparse_ratio") for row in group]
        ys = [f(row, "true_ce_nll") for row in group]
        ax.scatter(xs, ys, s=34, label=source)
    ax.set_xlabel("sparse_bf16 ratio")
    ax.set_ylabel("Measured CE/NLL")
    ax.set_title("MIRROR sparse_bf16: NLL vs Compression Ratio")
    ax.grid(True, color="#e5e7eb")
    ax.legend(loc="best")
    save_plot(fig, out_dir / "nll_vs_sparse_ratio")


def plot_predicted_vs_true(rows: list[dict[str, Any]], out_dir: Path) -> None:
    filtered = [row for row in rows if row.get("true_nll_delta_vs_bf16_clipped", "") != ""]
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    for source, group in group_by(filtered, "source").items():
        ax.scatter([f(row, "predicted_quality_cost") for row in group], [f(row, "true_nll_delta_vs_bf16_clipped") for row in group], s=34, label=source)
    max_v = max([f(row, "predicted_quality_cost") for row in filtered] + [f(row, "true_nll_delta_vs_bf16_clipped") for row in filtered] + [0.01])
    ax.plot([0, max_v], [0, max_v], color="#111827", linestyle=":", linewidth=1.0, label="ideal")
    ax.set_xlabel("Predicted quality_cost")
    ax.set_ylabel("Measured NLL delta vs dense_bf16 (clipped)")
    ax.set_title("Predicted vs Measured sparse_bf16 Quality Cost")
    ax.grid(True, color="#e5e7eb")
    ax.legend(loc="best")
    save_plot(fig, out_dir / "predicted_cost_vs_true_nll_delta")


def plot_residual_vs_ratio(rows: list[dict[str, Any]], out_dir: Path) -> None:
    filtered = [row for row in rows if row.get("quality_residual_vs_bf16", "") != ""]
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    for source, group in group_by(filtered, "source").items():
        ax.scatter([f(row, "sparse_ratio") for row in group], [f(row, "quality_residual_vs_bf16") for row in group], s=34, label=source)
    ax.axhline(0, color="#111827", linestyle=":", linewidth=1.0)
    ax.set_xlabel("sparse_bf16 ratio")
    ax.set_ylabel("Measured clipped NLL delta - predicted quality_cost")
    ax.set_title("Additive Model Residual vs sparse_bf16 Ratio")
    ax.grid(True, color="#e5e7eb")
    ax.legend(loc="best")
    save_plot(fig, out_dir / "residual_vs_sparse_ratio")


def plot_full_policy_comparison(rows: list[dict[str, Any]], out_dir: Path) -> None:
    labels = {"uniform_sparse_bf16", "batch_16_point_153", "batch_16_point_158", "batch_16_point_162", "sparse_bf16_lowerr_ratio_1"}
    selected = [row for row in rows if row["label"] in labels or row["key"] in labels]
    selected = sorted(selected, key=lambda row: f(row, "sparse_ratio"))
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    xs = range(len(selected))
    ax.bar([x - 0.18 for x in xs], [f(row, "predicted_quality_cost") for row in selected], width=0.36, label="predicted quality_cost")
    ax.bar([x + 0.18 for x in xs], [f(row, "true_nll_delta_vs_bf16_clipped") for row in selected], width=0.36, label="measured NLL delta vs bf16")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([short_label(row) for row in selected], rotation=25, ha="right")
    ax.set_ylabel("Cost / NLL delta")
    ax.set_title("Key sparse_bf16 Policies: Additive Prediction vs Measurement")
    ax.grid(True, axis="y", color="#e5e7eb")
    ax.legend(loc="best")
    save_plot(fig, out_dir / "key_policy_prediction_gap")


def plot_same_count_variance(rows: list[dict[str, Any]], out_dir: Path) -> None:
    controlled = [row for row in rows if row["source"] == "controlled_genimage_partial" and int(f(row, "count_sparse_bf16")) > 0]
    if not controlled:
        return
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    xs = [int(f(row, "count_sparse_bf16")) for row in controlled]
    ys = [f(row, "true_ce_nll") for row in controlled]
    colors = ["#dc2626" if row["label"].startswith("random") else "#2563eb" if row["label"].startswith(("lowerr", "speed")) else "#7c3aed" for row in controlled]
    ax.scatter(xs, ys, c=colors, s=36)
    ax.set_xlabel("sparse_bf16 module count")
    ax.set_ylabel("Measured GenImage partial CE/NLL")
    ax.set_title("Same-count sparse_bf16 Policies Have Large NLL Variance")
    ax.grid(True, color="#e5e7eb")
    save_plot(fig, out_dir / "same_count_nll_variance")


def short_label(row: dict[str, Any]) -> str:
    if row.get("key"):
        return row["key"].replace("batch_16_", "")
    return row["label"].replace("uniform_", "u_").replace("sparse_bf16_", "sbf16_")


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get(key, ""))].append(row)
    return dict(out)


def save_plot(fig: Any, stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def write_summary(path: Path, measurements: list[dict[str, Any]], diagnostics: list[dict[str, Any]], type_rows: list[dict[str, Any]], bucket_rows: list[dict[str, Any]], count_variance_rows: list[dict[str, Any]], correlation_rows: list[dict[str, Any]]) -> None:
    full = [row for row in measurements if row["eval_scope"] == "ALL-full"]
    gen = [row for row in measurements if row["eval_scope"] == "GenImage-partial"]
    key_labels = {"uniform_sparse_bf16", "sparse_bf16_lowerr_ratio_1"}
    key_labels.update({row["label"] for row in full if str(row.get("key", "")).endswith(("153", "158", "162"))})
    lines = [
        "# MIRROR sparse_bf16 Additivity Debug",
        "",
        f"- measurements: {len(measurements)}",
        f"- GenImage partial rows: {len(gen)}",
        f"- full validation rows: {len(full)}",
        f"- controlled GenImage partial rows: {sum(1 for row in measurements if row['source'] == 'controlled_genimage_partial')}",
        "",
        "## Key Residuals",
        "",
    ]
    for row in diagnostics[:12]:
        lines.append(
            f"- {row['source']} `{row['label']}` ratio={float(row['sparse_ratio']):.3f} "
            f"pred={float(row['predicted_quality_cost']):.5f} true_delta={float(row['true_nll_delta_vs_bf16_clipped']):.5f} "
            f"residual={float(row['quality_residual_vs_bf16']):+.5f} bal={float(row['true_bal_acc']):.5f}"
        )
    lines.extend(["", "## Diagnosis", ""])
    lines.append("- The additive model over-penalizes high-ratio sparse_bf16 in several full-validation cases when compared with uniform sparse_bf16.")
    lines.append("- The relationship between sparse ratio and measured CE/NLL is not purely additive: mid/high sparse ratios show large residual changes that are not explained by summed local output error alone.")
    lines.append("- Mixed policies can be worse than uniform sparse_bf16 even when their predicted quality cost is lower, indicating a missing backend-consistency or interaction term.")
    if type_rows:
        top = type_rows[0]
        lines.append(f"- Largest mean residual by module type appears in `{top['group']}` policies: mean_residual={float(top['mean_residual']):+.5f}.")
    if bucket_rows:
        top = bucket_rows[0]
        lines.append(f"- Largest mean residual by layer bucket appears in `{top['group']}`: mean_residual={float(top['mean_residual']):+.5f}.")
    if count_variance_rows:
        top = max(count_variance_rows, key=lambda row: float(row["nll_range"]))
        lines.append(f"- Same-count variance is large: count={top['count_sparse_bf16']} has NLL range={float(top['nll_range']):.5f} between `{top['best_label']}` and `{top['worst_label']}`.")
    controlled_corr = [row for row in correlation_rows if row["source"] == "controlled_genimage_partial"]
    if controlled_corr:
        best = max(controlled_corr, key=lambda row: abs(float(row["spearman"])))
        pred = next((row for row in controlled_corr if row["feature"] == "predicted_quality_cost"), None)
        lines.append(f"- In controlled policies, best monotonic feature is `{best['feature']}` with Spearman={float(best['spearman']):+.3f}.")
        if pred:
            lines.append(f"- Existing predicted_quality_cost has controlled Spearman={float(pred['spearman']):+.3f}, RMSE={float(pred['rmse_linear_fit']):.5f}.")
    lines.extend(["", "## Suggested Fix Direction", ""])
    lines.append("- Replace plain additive sparse_bf16 cost with a policy-level model that includes count/ratio, selected layer/type distribution, and backend diversity.")
    lines.append("- Penalize mixed backend policies when measured residuals show they are worse than uniform sparse_bf16 at similar or lower predicted cost.")
    lines.append("- Keep same-count random policies as held-out validation before using the revised quality model for Pareto optimization.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
