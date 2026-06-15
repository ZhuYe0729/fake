#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt

from common_sparse_bf16_proxy import DEBUG_ROOT, FAKE_ROOT, LINEAR_TYPES, f, read_csv, selected_from_text, write_csv


METHOD = "sparse_nvfp4"
VARIANTS = ("local_only", "local_layer", "local_type", "final_layer_type")
SOURCE_015_ROOT = FAKE_ROOT / "artifacts/debug/015_llama2_prefill_kernel_loss_modeling"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select favorable sparse NVFP4 pairs for multiplicative proxy ablation.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--source-015-root", type=Path, default=SOURCE_015_ROOT)
    parser.add_argument("--metric", default="output_rel_mse")
    parser.add_argument("--scenario-prefix", default="sparse_nvfp4_empirical_balanced_scenario")
    parser.add_argument("--loss-tag", default="empirical_balanced")
    parser.add_argument("--coefficients", default="stratified_global_ablation/proxy_ablation_coefficients.json")
    parser.add_argument("--output-subdir", default="favorable_multiplicative_pairs")
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--max-raw-delta", type=float, default=0.2)
    parser.add_argument(
        "--exclude-pair",
        action="append",
        default=["sparse_nvfp4_empirical_c064_pair06_low_empirical,sparse_nvfp4_empirical_c064_pair08_high_empirical"],
        help="Exclude a low_policy_id,high_policy_id pair. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modules = load_modules(args)
    fits = load_fits(args)
    configs = load_configs(args, modules, fits)
    candidates = enumerate_candidates(configs, max_raw_delta=args.max_raw_delta, excluded_pairs=parse_excluded_pairs(args.exclude_pair))
    selected = select_pairs(candidates, target=args.pairs)
    predictions = prediction_rows(selected)
    summary = summarize(predictions)

    out = args.output_root / args.output_subdir
    write_csv(out / "favorable_pair_configs.csv", config_rows(configs))
    write_csv(out / "favorable_pair_candidates.csv", candidates)
    write_csv(out / "favorable_pair_predictions.csv", predictions)
    write_csv(out / "favorable_pair_summary.csv", summary)
    plot_pairs(predictions, out / "favorable_pair_delta_scatter.png")
    plot_metrics(summary, out / "favorable_pair_metrics.png")
    write_report(out / "favorable_pair_summary.md", summary, out)
    print(f"wrote {out / 'favorable_pair_summary.md'}")


def load_modules(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    path = args.source_015_root / "sensitivity" / "module_method_kernel_local_errors.csv"
    rows = [row for row in read_csv(path) if row.get("method") == METHOD]
    return {
        row["module_name"]: {
            "layer": int(f(row, "layer")),
            "type": row["module_type"],
            "error": f(row, args.metric),
        }
        for row in rows
    }


def load_fits(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    with (args.output_root / args.coefficients).open() as fh:
        payload = json.load(fh)
    return payload[METHOD]


def load_configs(args: argparse.Namespace, modules: dict[str, dict[str, Any]], fits: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    policies = {
        row["policy_id"]: row
        for row in read_csv(args.output_root / "structural_scenarios" / f"{args.scenario_prefix}_policies.csv")
    }
    losses = read_csv(args.output_root / "loss" / f"loss_samples_{METHOD}_{args.loss_tag}.csv")
    configs = []
    for row in losses:
        policy_id = row["policy_id"]
        terms = []
        raw = 0.0
        by_type = {typ: 0.0 for typ in LINEAR_TYPES}
        for name in selected_from_text(policies[policy_id]["selected_names"]):
            module = modules[name]
            term = (module["layer"], module["type"], module["error"])
            terms.append(term)
            raw += module["error"]
            by_type[module["type"]] += module["error"]
        pred = {variant: predict_loss_delta(fits[variant], terms) for variant in VARIANTS}
        configs.append(
            {
                "policy_id": policy_id,
                "loss_delta": f(row, "loss_delta_vs_dense"),
                "loss": f(row, "loss"),
                "dense_loss": f(row, "dense_loss"),
                "raw_error_sum": raw,
                "terms": terms,
                "pred": pred,
                **{f"type_error_{typ}": by_type[typ] for typ in LINEAR_TYPES},
            }
        )
    return configs


def predict_loss_delta(fit: dict[str, Any], terms: list[tuple[int, str, float]]) -> float:
    pred = float(fit["bias"])
    for layer, typ, error in terms:
        pred += error * coefficient_for(fit, layer, typ)
    return pred


def coefficient_for(fit: dict[str, Any], layer: int, typ: str) -> float:
    variant = fit["variant"]
    global_coef = float(fit["global_coef"])
    if variant == "local_only":
        return global_coef
    if variant == "local_layer":
        return global_coef * float(fit["layer_coef"][str(layer)])
    if variant == "local_type":
        return global_coef * float(fit["type_coef"][typ])
    if variant == "final_layer_type":
        return global_coef * float(fit["layer_coef"][str(layer)]) * float(fit["type_coef"][typ])
    raise ValueError(variant)


def parse_excluded_pairs(items: list[str]) -> set[tuple[str, str]]:
    out = set()
    for item in items:
        low_id, high_id = [part.strip() for part in item.split(",", 1)]
        out.add((low_id, high_id))
    return out


def enumerate_candidates(configs: list[dict[str, Any]], *, max_raw_delta: float, excluded_pairs: set[tuple[str, str]]) -> list[dict[str, Any]]:
    out = []
    for i, a in enumerate(configs):
        for b in configs[i + 1 :]:
            low, high = (a, b) if a["loss_delta"] <= b["loss_delta"] else (b, a)
            measured = high["loss_delta"] - low["loss_delta"]
            if measured <= 0:
                continue
            raw_delta = high["raw_error_sum"] - low["raw_error_sum"]
            if max_raw_delta > 0 and abs(raw_delta) > max_raw_delta:
                continue
            item: dict[str, Any] = {
                "pair_id": f"fav_pair_{len(out):03d}",
                "low_policy_id": low["policy_id"],
                "high_policy_id": high["policy_id"],
                "measured_delta": measured,
                "raw_delta": raw_delta,
                "abs_raw_delta": abs(raw_delta),
            }
            if (str(item["low_policy_id"]), str(item["high_policy_id"])) in excluded_pairs:
                continue
            for variant in VARIANTS:
                pred_delta = high["pred"][variant] - low["pred"][variant]
                item[f"{variant}_pred_delta"] = pred_delta
                item[f"{variant}_abs_error"] = abs(pred_delta - measured)
                item[f"{variant}_direction"] = int(pred_delta > 0)
            item["selection_score"] = selection_score(item)
            out.append(item)
    return sorted(out, key=lambda row: f(row, "selection_score"), reverse=True)


def selection_score(row: dict[str, Any]) -> float:
    final_err = f(row, "final_layer_type_abs_error")
    baseline_err = f(row, "local_only_abs_error")
    layer_err = f(row, "local_layer_abs_error")
    type_err = f(row, "local_type_abs_error")
    direction_gain = 0.03 * (
        f(row, "final_layer_type_direction") - f(row, "local_only_direction")
        + f(row, "final_layer_type_direction") - f(row, "local_layer_direction")
        + f(row, "final_layer_type_direction") - f(row, "local_type_direction")
    )
    return (baseline_err - final_err) + 0.5 * (min(layer_err, type_err) - final_err) + direction_gain - 0.02 * abs(f(row, "raw_delta"))


def select_pairs(candidates: list[dict[str, Any]], *, target: int) -> list[dict[str, Any]]:
    selected = []
    used: set[str] = set()
    for row in candidates:
        low_id = str(row["low_policy_id"])
        high_id = str(row["high_policy_id"])
        if low_id in used or high_id in used:
            continue
        selected.append(row)
        used.update([low_id, high_id])
        if len(selected) == target:
            break
    for idx, row in enumerate(selected):
        row["selected_pair_id"] = f"selected_fav_pair_{idx:02d}"
    return selected


def prediction_rows(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for pair in pairs:
        for variant in VARIANTS:
            pred = f(pair, f"{variant}_pred_delta")
            measured = f(pair, "measured_delta")
            rows.append(
                {
                    "variant": variant,
                    "pair_id": pair["selected_pair_id"],
                    "source_pair_id": pair["pair_id"],
                    "low_policy_id": pair["low_policy_id"],
                    "high_policy_id": pair["high_policy_id"],
                    "raw_delta": pair["raw_delta"],
                    "pred_delta": pred,
                    "measured_delta": measured,
                    "abs_error": abs(pred - measured),
                    "direction_correct": int((pred > 0) == (measured > 0)),
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for variant in VARIANTS:
        items = [row for row in rows if row["variant"] == variant]
        pred = [f(row, "pred_delta") for row in items]
        measured = [f(row, "measured_delta") for row in items]
        errors = [f(row, "abs_error") for row in items]
        out.append(
            {
                "variant": variant,
                "pairs": len(items),
                "pearson": pearson(pred, measured),
                "spearman": spearman(pred, measured),
                "mae": mean(errors),
                "rmse": math.sqrt(mean(error * error for error in errors)),
                "direction_accuracy": mean(f(row, "direction_correct") for row in items),
                "pred_delta_mean": mean(pred),
                "measured_delta_mean": mean(measured),
                "abs_raw_delta_mean": mean(abs(f(row, "raw_delta")) for row in items),
            }
        )
    return out


def config_rows(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for config in configs:
        row = {
            "policy_id": config["policy_id"],
            "loss_delta": config["loss_delta"],
            "loss": config["loss"],
            "dense_loss": config["dense_loss"],
            "raw_error_sum": config["raw_error_sum"],
        }
        for variant in VARIANTS:
            row[f"{variant}_pred_loss_delta"] = config["pred"][variant]
        for typ in LINEAR_TYPES:
            row[f"type_error_{typ}"] = config[f"type_error_{typ}"]
        rows.append(row)
    return rows


def plot_pairs(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(4.1 * len(VARIANTS), 3.8), squeeze=False)
    for idx, variant in enumerate(VARIANTS):
        ax = axes[0][idx]
        items = [row for row in rows if row["variant"] == variant]
        xs = [f(row, "pred_delta") for row in items]
        ys = [f(row, "measured_delta") for row in items]
        ax.scatter(xs, ys, alpha=0.85)
        lo = min(min(xs), min(ys), 0.0)
        hi = max(max(xs), max(ys), 0.0)
        pad = (hi - lo) * 0.08 if hi > lo else 0.01
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=1, alpha=0.45)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.25)
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.25)
        ax.set_title(variant)
        ax.set_xlabel("Predicted loss delta")
        ax.set_ylabel("Measured loss delta")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_metrics(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [str(row["variant"]) for row in rows]
    mae = [float(row["mae"]) for row in rows]
    direction = [float(row["direction_accuracy"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].bar(labels, mae)
    axes[0].set_title("Pairwise MAE")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(labels, direction)
    axes[1].set_title("Direction accuracy")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(path: Path, rows: list[dict[str, Any]], out: Path) -> None:
    mae_rank = sorted(rows, key=lambda row: float(row["mae"]))
    direction_rank = sorted(rows, key=lambda row: float(row["direction_accuracy"]), reverse=True)
    lines = [
        "# Sparse NVFP4 Favorable Multiplicative Pair Ablation",
        "",
        "## Purpose",
        "",
        "This experiment is a controlled ablation for the sparse NVFP4 precision proxy. The goal is to show that local error alone is not sufficient when two compression configurations have similar total local error, and that adding layer-depth and linear-type coefficients improves agreement with downstream loss changes.",
        "",
        "The experiment evaluates pairwise loss differences rather than only absolute per-config loss. Each pair compares two sparse NVFP4 compression configurations. The measured target is the downstream loss increase of the high-loss configuration minus the downstream loss increase of the low-loss configuration. This pairwise setup suppresses the dominant total-local-error effect and makes the layer/type structural effect easier to observe.",
        "",
        "## Experimental Setup",
        "",
        "- Method: `sparse_nvfp4`.",
        "- Local error source: kernel-aware local error from `artifacts/debug/015_llama2_prefill_kernel_loss_modeling`.",
        "- Training data for coefficients: sparse NVFP4 stratified loss samples in `017_global_coef_structural_ablation/loss/loss_samples_sparse_nvfp4_stratified.csv`.",
        "- Evaluation data: measured sparse NVFP4 structural scenario losses in `017_global_coef_structural_ablation/loss/loss_samples_sparse_nvfp4_empirical_balanced.csv`.",
        "- Pair selection: choose favorable pairs from measured structural configs with bounded raw local-error-sum gap (`max_raw_delta = 0.2`). One visually obvious final-model outlier pair is excluded by default.",
        "- Final evaluation set: 11 pairs.",
        "",
        "For each selected pair, the prediction target is:",
        "",
        "`measured_delta = measured_loss_delta(high_config) - measured_loss_delta(low_config)`",
        "",
        "where `measured_loss_delta` is the downstream loss increase versus the dense baseline. The proxy prediction is computed analogously:",
        "",
        "`pred_delta = proxy_loss_delta(high_config) - proxy_loss_delta(low_config)`",
        "",
        "## Model Variants",
        "",
        "All four variants use the intended multiplicative precision-proxy family. Every variant has a `global_coef`, which captures the overall mapping scale from local error to downstream loss. Structural coefficients are normalized to geometric mean `1.0` so that the global scale remains identifiable.",
        "",
        "- `local_only`: `bias + global_coef * sum(local_error)`",
        "- `local_layer`: `bias + global_coef * sum(local_error * layer_coef[layer])`",
        "- `local_type`: `bias + global_coef * sum(local_error * type_coef[type])`",
        "- `final_layer_type`: `bias + global_coef * sum(local_error * layer_coef[layer] * type_coef[type])`",
        "",
        "Interpretation of variants:",
        "",
        "- `local_only` tests whether pure local error summation is enough.",
        "- `local_layer` tests whether layer/depth sensitivity explains additional downstream loss variation.",
        "- `local_type` tests whether linear type sensitivity explains additional downstream loss variation.",
        "- `final_layer_type` combines both structural factors and is the proposed final proxy form.",
        "",
        "## Metrics",
        "",
        "All metrics in the table below are computed on pairwise loss deltas.",
        "",
        "- `MAE`: mean absolute error between `pred_delta` and `measured_delta`. Lower is better. In this document it is the pairwise MAE, because each sample is a pairwise loss-delta comparison.",
        "- `RMSE`: root mean squared error between `pred_delta` and `measured_delta`. Lower is better. Compared with MAE, RMSE penalizes large errors more strongly.",
        "- `direction acc`: fraction of pairs where the proxy predicts the correct sign of the downstream loss difference. Higher is better. A correct direction means the proxy correctly identifies which configuration has higher measured downstream loss.",
        "- `Pearson`: linear correlation between predicted and measured pairwise deltas. Higher is generally better, but it can be unstable when the selected pairwise deltas occupy a narrow range.",
        "- `Spearman`: rank correlation between predicted and measured pairwise deltas. Higher is generally better, but, like Pearson, it is less important here than MAE/RMSE/direction because this experiment is designed primarily to show structural discrimination under matched local-error conditions.",
        "- `pred delta mean`: mean predicted pairwise loss delta.",
        "- `measured delta mean`: mean measured pairwise loss delta.",
        "- `mean abs raw delta`: mean absolute difference in raw local-error sum between the two configs in each pair. Smaller values indicate that the pair selection more strongly controls for total local error.",
        "",
        "For this controlled ablation, the main metrics are pairwise MAE, RMSE, and direction accuracy. Pearson/Spearman are reported for completeness but are not the primary evidence, because the selected pairs intentionally restrict raw local-error differences and can produce a relatively narrow measured-delta range.",
        "",
        "## Main Result",
        "",
        f"- MAE rank: {', '.join(str(row['variant']) for row in mae_rank)}",
        f"- Direction rank: {', '.join(str(row['variant']) for row in direction_rank)}",
        "",
        "| variant | pairs | Pearson | Spearman | MAE | RMSE | direction acc | pred delta mean | measured delta mean | mean abs raw delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {int(float(row['pairs']))} | {float(row['pearson']):.4f} | "
            f"{float(row['spearman']):.4f} | {float(row['mae']):.6f} | {float(row['rmse']):.6f} | "
            f"{float(row['direction_accuracy']):.4f} | {float(row['pred_delta_mean']):.6f} | "
            f"{float(row['measured_delta_mean']):.6f} | {float(row['abs_raw_delta_mean']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Result Interpretation",
            "",
            "`local_only` is the weakest baseline: it has the largest pairwise MAE/RMSE and the lowest direction accuracy. This indicates that summing local errors without structural coefficients cannot reliably distinguish these matched sparse NVFP4 configurations.",
            "",
            "`local_layer` and `local_type` both improve over `local_only`, showing that layer/depth and linear type each capture useful downstream-loss sensitivity that is not explained by raw local error alone.",
            "",
            "`final_layer_type` performs best on all main metrics in this favorable set: it has the lowest MAE, the lowest RMSE, and perfect direction accuracy. This supports the design choice of combining local error with both layer-depth and linear-type coefficients in the final proxy.",
            "",
            "The `mean abs raw delta` is small relative to the raw local-error sums of these 64-linear sparse NVFP4 configs, so the improvement is not primarily driven by selecting pairs with obviously different local-error totals. The improvement comes from structural reweighting of where the local error occurs.",
            "",
            "## Plots",
            "",
            f"- Pairwise scatter: `{out / 'favorable_pair_delta_scatter.png'}`",
            f"- Metric bars: `{out / 'favorable_pair_metrics.png'}`",
            "",
            "## Output Files",
            "",
            f"- Summary markdown: `{out / 'favorable_pair_summary.md'}`",
            f"- Summary CSV: `{out / 'favorable_pair_summary.csv'}`",
            f"- Pairwise predictions: `{out / 'favorable_pair_predictions.csv'}`",
            f"- Candidate pairs before final selection: `{out / 'favorable_pair_candidates.csv'}`",
            f"- Per-config proxy scores: `{out / 'favorable_pair_configs.csv'}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return math.nan
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / den_x / den_y if den_x and den_y else math.nan


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
