#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt

from common_fakevlm_pareto import DEBUG_ROOT, parse_batches, read_csv, read_json, write_csv, write_json


BACKENDS = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4")
SUFFIX = "_prediction_vs_actual"
LOSS_DEFINITION = "assistant_answer_token_nll_v2_active_prefix_aligned"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FakeVLM prediction-versus-actual validation artifacts.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--speed-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batches", default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_root
    repo_root = root.parents[2]
    speed_root = args.speed_root or repo_root / "artifacts/debug/021_fakevlm_linear_hybrid_prefill_speed"
    batches = parse_batches(args.batches)
    out_dir = args.output_dir or root / "prediction_vs_actual"
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = filter_batches(read_csv(root / "validation/selected_pareto_points.csv"), batches)
    selected_keys = {policy_key(integer(row, "batch_size"), integer(row, "point_index")) for row in selected}
    speed_rows = [row for row in read_csv(root / "validation/pareto_speed_validation.csv") if row["key"] in selected_keys]
    accuracy_rows = [row for row in read_csv(root / "quality/validation_quality.csv") if row["key"] in selected_keys]
    loss_rows = [row for row in read_csv(root / "quality/validation_loss.csv") if row["key"] in selected_keys]
    expected = len(selected)
    if expected == 0:
        raise RuntimeError(f"no selected policies for batches={batches}")
    require_count("speed validation", speed_rows, expected)
    require_count("accuracy validation", accuracy_rows, expected)
    require_count("loss validation", loss_rows, expected)

    quality_rows = build_quality_rows(root, selected, accuracy_rows, loss_rows)
    linear_rows, predicted_maps, actual_maps, source_maps = build_linear_rows(speed_root, batches)
    e2e_rows = build_e2e_rows(selected, speed_rows, predicted_maps, actual_maps, source_maps, batches)

    write_csv(out_dir / f"quality_comparison{SUFFIX}.csv", quality_rows)
    write_csv(out_dir / f"single_linear_latency_comparison{SUFFIX}.csv", linear_rows)
    write_csv(out_dir / f"e2e_latency_comparison{SUFFIX}.csv", e2e_rows)

    metrics = {
        "quality": quality_metrics(quality_rows),
        "single_linear": grouped_latency_metrics(linear_rows, "predicted_latency_ms", "actual_latency_ms", "prediction_kind"),
        "e2e": grouped_e2e_metrics(e2e_rows, batches),
    }
    write_json(out_dir / f"comparison_metrics{SUFFIX}.json", metrics)
    write_summary(out_dir, quality_rows, linear_rows, e2e_rows, metrics, batches)
    plot_quality(out_dir, quality_rows, batches)
    plot_single_linear(out_dir, linear_rows, batches)
    plot_e2e(out_dir, e2e_rows, batches)
    print(f"wrote prediction comparison: quality={len(quality_rows)} linear={len(linear_rows)} e2e={len(e2e_rows)}")


def build_quality_rows(
    root: Path,
    selected: list[dict[str, str]],
    accuracy_rows: list[dict[str, str]],
    loss_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    accuracy_by_key = {row["key"]: row for row in accuracy_rows}
    loss_by_key = {row["key"]: row for row in loss_rows}
    invalid_definitions = sorted({row.get("loss_definition", "") for row in loss_rows if row.get("loss_definition", "") != LOSS_DEFINITION})
    if invalid_definitions:
        raise RuntimeError(
            f"loss comparison rows use invalid definitions {invalid_definitions}; expected {LOSS_DEFINITION}"
        )
    metadata = read_json(root / "global_coefficients/proxy_ablation_metadata.json")
    model_dense_nll = float(metadata["dense_nll"])
    out = []
    for row in sorted(selected, key=lambda item: (integer(item, "batch_size"), integer(item, "point_index"))):
        batch = integer(row, "batch_size")
        point = integer(row, "point_index")
        key = policy_key(batch, point)
        loss = required(loss_by_key, key, "loss")
        accuracy = required(accuracy_by_key, key, "accuracy")
        predicted_delta = number(row, "quality_cost")
        actual_nll = number(loss, "nll")
        raw_delta = actual_nll - model_dense_nll
        predicted_nll = model_dense_nll + predicted_delta
        out.append(
            {
                "key": key,
                "batch_size": batch,
                "point_index": point,
                "policy_json": row["policy_json"],
                "model_dense_nll": model_dense_nll,
                "predicted_nll": predicted_nll,
                "actual_nll": actual_nll,
                "predicted_nll_delta": predicted_delta,
                "actual_nll_delta_raw": raw_delta,
                "actual_nll_delta_clipped": max(0.0, raw_delta),
                "nll_prediction_error": predicted_nll - actual_nll,
                "nll_prediction_abs_error": abs(predicted_nll - actual_nll),
                "fakeclue_global_accuracy": number(accuracy, "global_accuracy"),
                "fakeclue_total_right": integer(accuracy, "total_right"),
                "fakeclue_total_wrong": integer(accuracy, "total_wrong"),
                "loss_tokens": integer(loss, "loss_tokens"),
                "loss_definition": loss["loss_definition"],
            }
        )
    return out


def build_linear_rows(
    speed_root: Path,
    batches: tuple[int, ...],
) -> tuple[
    list[dict[str, Any]],
    dict[int, dict[tuple[int, int, str], float]],
    dict[int, dict[tuple[int, int, str], float]],
    dict[int, dict[tuple[int, int, str], str]],
]:
    rows = []
    predicted_maps = {}
    actual_maps = {}
    source_maps = {}
    for batch in batches:
        predicted = read_csv(speed_root / "candidates/latency_model" / f"batch_{batch}.csv")
        actual = read_csv(speed_root / "candidates/manual_profile" / f"batch_{batch}.csv")
        predicted_by_key = {linear_key(row): row for row in predicted if truthy(row.get("supported", "True"))}
        actual_by_key = {linear_key(row): row for row in actual if truthy(row.get("supported", "True"))}
        if set(predicted_by_key) != set(actual_by_key):
            raise RuntimeError(f"batch {batch} predicted/actual linear keys differ")
        predicted_maps[batch] = {}
        actual_maps[batch] = {}
        source_maps[batch] = {}
        for key in sorted(predicted_by_key):
            pred = predicted_by_key[key]
            real = actual_by_key[key]
            pred_ms = number(pred, "latency_ms")
            real_ms = number(real, "latency_ms")
            n, k, backend = key
            kind = "measured_lookup" if pred.get("source") == "measured" else "model_prediction"
            predicted_maps[batch][key] = pred_ms
            actual_maps[batch][key] = real_ms
            source_maps[batch][key] = kind
            rows.append(
                {
                    "batch_size": batch,
                    "m": integer(pred, "m"),
                    "n": n,
                    "k": k,
                    "backend": backend,
                    "prediction_kind": kind,
                    "predictor_source": pred.get("source", ""),
                    "predicted_latency_ms": pred_ms,
                    "actual_latency_ms": real_ms,
                    "error_ms": pred_ms - real_ms,
                    "abs_error_ms": abs(pred_ms - real_ms),
                    "relative_error": (pred_ms - real_ms) / real_ms,
                    "abs_relative_error": abs(pred_ms - real_ms) / real_ms,
                    "prediction_status": pred.get("prediction_status", ""),
                }
            )
    require_count("single-linear comparison", rows, 12 * len(batches))
    return rows, predicted_maps, actual_maps, source_maps


def build_e2e_rows(
    selected: list[dict[str, str]],
    speed_rows: list[dict[str, str]],
    predicted_maps: dict[int, dict[tuple[int, int, str], float]],
    actual_maps: dict[int, dict[tuple[int, int, str], float]],
    source_maps: dict[int, dict[tuple[int, int, str], str]],
    batches: tuple[int, ...],
) -> list[dict[str, Any]]:
    speed_by_key = {row["key"]: row for row in speed_rows}
    policy_data = {}
    for row in selected:
        batch = integer(row, "batch_size")
        point = integer(row, "point_index")
        modules = read_json(Path(row["policy_json"]))["modules"]
        if len(modules) != 224:
            raise RuntimeError(f"{policy_key(batch, point)} has {len(modules)} modules, expected 224")
        policy_data[(batch, point)] = modules

    nonlinear_by_batch = {}
    for batch in batches:
        dense_speed = required(speed_by_key, policy_key(batch, 0), "dense speed")
        dense_modules = policy_data[(batch, 0)]
        dense_actual_linear = sum_policy_latency(dense_modules, actual_maps[batch])
        nonlinear_by_batch[batch] = number(dense_speed, "e2e_prefill_mean_ms") - dense_actual_linear

    out = []
    for row in sorted(selected, key=lambda item: (integer(item, "batch_size"), integer(item, "point_index"))):
        batch = integer(row, "batch_size")
        point = integer(row, "point_index")
        key = policy_key(batch, point)
        measured = required(speed_by_key, key, "speed")
        modules = policy_data[(batch, point)]
        predicted_linear = sum_policy_latency(modules, predicted_maps[batch])
        actual_linear = sum_policy_latency(modules, actual_maps[batch])
        nonlinear = nonlinear_by_batch[batch]
        predicted_e2e = nonlinear + predicted_linear
        oracle_linear_e2e = nonlinear + actual_linear
        actual_e2e = number(measured, "e2e_prefill_mean_ms")
        source_counts = {"measured_lookup": 0, "model_prediction": 0}
        for module in modules:
            shape = module_linear_key(module)
            source_counts[source_maps[batch][shape]] += 1
        out.append(
            {
                "key": key,
                "batch_size": batch,
                "point_index": point,
                "policy_json": row["policy_json"],
                "predicted_linear_latency_ms": predicted_linear,
                "optimizer_linear_latency_ms": number(row, "latency_ms"),
                "actual_linear_latency_ms": actual_linear,
                "measured_nonlinear_latency_ms": nonlinear,
                "predicted_e2e_latency_ms": predicted_e2e,
                "oracle_linear_e2e_latency_ms": oracle_linear_e2e,
                "actual_e2e_latency_ms": actual_e2e,
                "e2e_prediction_error_ms": predicted_e2e - actual_e2e,
                "e2e_prediction_abs_error_ms": abs(predicted_e2e - actual_e2e),
                "e2e_prediction_relative_error": (predicted_e2e - actual_e2e) / actual_e2e,
                "oracle_linear_e2e_error_ms": oracle_linear_e2e - actual_e2e,
                "predictor_measured_lookup_modules": source_counts["measured_lookup"],
                "predictor_model_prediction_modules": source_counts["model_prediction"],
            }
        )
    return out


def sum_policy_latency(modules: list[dict[str, Any]], latency_map: dict[tuple[int, int, str], float]) -> float:
    total = 0.0
    for module in modules:
        key = module_linear_key(module)
        if key not in latency_map:
            raise RuntimeError(f"missing latency for {key}")
        total += latency_map[key]
    return total


def module_linear_key(module: dict[str, Any]) -> tuple[int, int, str]:
    return int(module["n"]), int(module["k"]), str(module.get("selected_method") or module["backend"])


def linear_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return integer(row, "n"), integer(row, "k"), str(row["backend"])


def quality_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_deltas = [float(row["actual_nll_delta_raw"]) for row in rows]
    out = {
        "absolute_nll": metric_block(rows, "predicted_nll", "actual_nll"),
        "clipped_nll_delta": metric_block(rows, "predicted_nll_delta", "actual_nll_delta_clipped"),
        "raw_nll_delta": metric_block(rows, "predicted_nll_delta", "actual_nll_delta_raw"),
        "actual_raw_delta_counts": {
            "negative": sum(value < -1e-12 for value in raw_deltas),
            "zero": sum(abs(value) <= 1e-12 for value in raw_deltas),
            "positive": sum(value > 1e-12 for value in raw_deltas),
        },
    }
    out["predicted_cost_vs_accuracy"] = correlation_block(
        [float(row["predicted_nll_delta"]) for row in rows],
        [float(row["fakeclue_global_accuracy"]) for row in rows],
    )
    out["actual_nll_vs_accuracy"] = correlation_block(
        [float(row["actual_nll"]) for row in rows],
        [float(row["fakeclue_global_accuracy"]) for row in rows],
    )
    return out


def grouped_latency_metrics(rows: list[dict[str, Any]], predicted: str, actual: str, group: str) -> dict[str, Any]:
    out = {"all": metric_block(rows, predicted, actual)}
    for value in sorted({str(row[group]) for row in rows}):
        out[value] = metric_block([row for row in rows if row[group] == value], predicted, actual)
    return out


def grouped_e2e_metrics(rows: list[dict[str, Any]], batches: tuple[int, ...]) -> dict[str, Any]:
    out = {"all": metric_block(rows, "predicted_e2e_latency_ms", "actual_e2e_latency_ms")}
    out["oracle_linear_all"] = metric_block(rows, "oracle_linear_e2e_latency_ms", "actual_e2e_latency_ms")
    for batch in batches:
        batch_rows = [row for row in rows if int(row["batch_size"]) == batch]
        out[f"batch_{batch}"] = metric_block(batch_rows, "predicted_e2e_latency_ms", "actual_e2e_latency_ms")
    return out


def metric_block(rows: list[dict[str, Any]], predicted: str, actual: str) -> dict[str, float | int]:
    xs = [float(row[predicted]) for row in rows]
    ys = [float(row[actual]) for row in rows]
    errors = [x - y for x, y in zip(xs, ys)]
    return {
        "rows": len(rows),
        "mae": mean(abs(error) for error in errors),
        "rmse": math.sqrt(mean(error * error for error in errors)),
        "mape": mean(abs(error) / abs(y) for error, y in zip(errors, ys) if y != 0.0),
        **correlation_block(xs, ys),
    }


def correlation_block(xs: list[float], ys: list[float]) -> dict[str, float]:
    return {"pearson": pearson(xs, ys), "spearman": pearson(ranks(xs), ranks(ys))}


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return numerator / denominator if denominator else 0.0


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + end - 1) / 2.0 + 1.0
        for offset in range(index, end):
            out[indexed[offset][0]] = rank
        index = end
    return out


def plot_quality(out_dir: Path, rows: list[dict[str, Any]], batches: tuple[int, ...]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for batch in batches:
        batch_rows = [row for row in rows if int(row["batch_size"]) == batch]
        ax.scatter([row["actual_nll"] for row in batch_rows], [row["predicted_nll"] for row in batch_rows], label=f"batch {batch}", s=48)
    add_identity(ax, rows, "actual_nll", "predicted_nll")
    ax.set_xlabel("Actual assistant-answer NLL")
    ax.set_ylabel("Predicted assistant-answer NLL")
    ax.set_title("FakeVLM predicted vs actual NLL")
    ax.grid(True, color="#e5e7eb")
    ax.legend()
    save_figure(fig, out_dir / f"quality_nll_scatter{SUFFIX}")

    for batch in batches:
        batch_rows = [row for row in rows if int(row["batch_size"]) == batch]
        points = [int(row["point_index"]) for row in batch_rows]
        fig, axes = plt.subplots(3, 1, figsize=(9.5, 10), sharex=True)
        axes[0].plot(points, [row["predicted_nll"] for row in batch_rows], "o-", label="Predicted NLL")
        axes[0].plot(points, [row["actual_nll"] for row in batch_rows], "s-", label="Actual NLL")
        axes[0].set_ylabel("NLL")
        axes[0].legend()
        axes[1].plot(points, [row["predicted_nll_delta"] for row in batch_rows], "o-", label="Predicted delta")
        axes[1].plot(points, [row["actual_nll_delta_raw"] for row in batch_rows], "s-", label="Actual raw delta")
        axes[1].axhline(0.0, color="#6b7280", linewidth=1)
        axes[1].set_ylabel("NLL delta")
        axes[1].legend()
        axes[2].plot(points, [row["fakeclue_global_accuracy"] for row in batch_rows], "o-", color="#047857")
        axes[2].set_ylabel("FakeClue accuracy")
        axes[2].set_xlabel("Pareto point index")
        for ax in axes:
            ax.grid(True, color="#e5e7eb")
        fig.suptitle(f"FakeVLM quality prediction and downstream result: batch {batch}")
        fig.tight_layout()
        save_figure(fig, out_dir / f"quality_batch_{batch}{SUFFIX}")


def plot_single_linear(out_dir: Path, rows: list[dict[str, Any]], batches: tuple[int, ...]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    styles = {"measured_lookup": "o", "model_prediction": "s"}
    for kind, marker in styles.items():
        subset = [row for row in rows if row["prediction_kind"] == kind]
        ax.scatter([row["actual_latency_ms"] for row in subset], [row["predicted_latency_ms"] for row in subset], marker=marker, s=48, label=kind)
    add_identity(ax, rows, "actual_latency_ms", "predicted_latency_ms")
    ax.set_xlabel("Actual single-linear latency (ms)")
    ax.set_ylabel("Latency-model output (ms)")
    ax.set_title("FakeVLM single-linear predicted vs actual latency")
    ax.grid(True, color="#e5e7eb")
    ax.legend()
    save_figure(fig, out_dir / f"single_linear_scatter{SUFFIX}")

    for batch in batches:
        batch_rows = [row for row in rows if int(row["batch_size"]) == batch]
        labels = [f"{row['backend']}\n{row['n']}x{row['k']}" for row in batch_rows]
        positions = list(range(len(batch_rows)))
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.bar([x - 0.2 for x in positions], [row["actual_latency_ms"] for row in batch_rows], width=0.4, label="Actual manual profile")
        ax.bar([x + 0.2 for x in positions], [row["predicted_latency_ms"] for row in batch_rows], width=0.4, label="Latency-model output")
        ax.set_xticks(positions, labels, rotation=40, ha="right")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"FakeVLM single-linear latency comparison: batch {batch}")
        ax.grid(True, axis="y", color="#e5e7eb")
        ax.legend()
        fig.tight_layout()
        save_figure(fig, out_dir / f"single_linear_batch_{batch}{SUFFIX}")


def plot_e2e(out_dir: Path, rows: list[dict[str, Any]], batches: tuple[int, ...]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for batch in batches:
        batch_rows = [row for row in rows if int(row["batch_size"]) == batch]
        ax.scatter([row["actual_e2e_latency_ms"] for row in batch_rows], [row["predicted_e2e_latency_ms"] for row in batch_rows], label=f"batch {batch}", s=48)
    add_identity(ax, rows, "actual_e2e_latency_ms", "predicted_e2e_latency_ms")
    ax.set_xlabel("Actual E2E prefill latency (ms)")
    ax.set_ylabel("Predicted E2E prefill latency (ms)")
    ax.set_title("FakeVLM predicted vs actual E2E prefill latency")
    ax.grid(True, color="#e5e7eb")
    ax.legend()
    save_figure(fig, out_dir / f"e2e_latency_scatter{SUFFIX}")

    for batch in batches:
        batch_rows = [row for row in rows if int(row["batch_size"]) == batch]
        points = [int(row["point_index"]) for row in batch_rows]
        fig, ax = plt.subplots(figsize=(9.5, 6.5))
        ax.plot(points, [row["actual_e2e_latency_ms"] for row in batch_rows], "o-", label="Actual E2E")
        ax.plot(points, [row["predicted_e2e_latency_ms"] for row in batch_rows], "s-", label="Predicted E2E")
        ax.plot(points, [row["oracle_linear_e2e_latency_ms"] for row in batch_rows], "^-", label="Measured-linear diagnostic")
        ax.set_xlabel("Pareto point index")
        ax.set_ylabel("Prefill latency (ms)")
        ax.set_title(f"FakeVLM E2E prediction comparison: batch {batch}")
        ax.grid(True, color="#e5e7eb")
        ax.legend()
        fig.tight_layout()
        save_figure(fig, out_dir / f"e2e_latency_batch_{batch}{SUFFIX}")


def add_identity(ax: Any, rows: list[dict[str, Any]], actual: str, predicted: str) -> None:
    values = [float(row[actual]) for row in rows] + [float(row[predicted]) for row in rows]
    low, high = min(values), max(values)
    margin = max((high - low) * 0.05, 1e-6)
    ax.plot([low - margin, high + margin], [low - margin, high + margin], "--", color="#111827", linewidth=1, label="Ideal")


def save_figure(fig: Any, path_without_suffix: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=220)
    fig.savefig(path_without_suffix.with_suffix(".pdf"))
    plt.close(fig)


def write_summary(
    out_dir: Path,
    quality_rows: list[dict[str, Any]],
    linear_rows: list[dict[str, Any]],
    e2e_rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    batches: tuple[int, ...],
) -> None:
    lines = [
        "# FakeVLM Prediction vs Actual Summary",
        "",
        f"- Quality policies: {len(quality_rows)}",
        f"- Single-linear comparisons: {len(linear_rows)}",
        f"- E2E policy comparisons: {len(e2e_rows)}",
        "- Predicted quality is NLL-based; no downstream accuracy prediction is inferred from NLL cost.",
        "",
        "## Quality",
        "",
        "| Comparison | MAE | RMSE | MAPE | Pearson | Spearman |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("Absolute NLL", "absolute_nll"), ("Clipped NLL delta", "clipped_nll_delta"), ("Raw NLL delta", "raw_nll_delta")):
        lines.append(metric_markdown_row(label, metrics["quality"][key]))
    counts = metrics["quality"]["actual_raw_delta_counts"]
    lines.extend(
        [
            "",
            f"Actual raw NLL deltas: {counts['negative']} negative, {counts['zero']} zero, {counts['positive']} positive.",
        ]
    )
    if metrics["quality"]["absolute_nll"]["pearson"] < 0:
        lines.append(
            "WARNING: predicted and actual NLL are negatively correlated on selected policies; "
            "the fitted NLL cost does not generalize as a calibrated selected-policy loss predictor."
        )
    lines.extend(["", "## Single Linear", "", "| Source kind | Rows | MAE ms | RMSE ms | MAPE | Pearson | Spearman |", "|---|---:|---:|---:|---:|---:|---:|"])
    for key in ("all", "measured_lookup", "model_prediction"):
        block = metrics["single_linear"][key]
        lines.append(f"| {key} | {block['rows']} | {block['mae']:.6f} | {block['rmse']:.6f} | {block['mape']:.6f} | {block['pearson']:.6f} | {block['spearman']:.6f} |")
    lines.extend(["", "## End To End", "", "| Batch | MAE ms | RMSE ms | MAPE | Pearson | Spearman |", "|---:|---:|---:|---:|---:|---:|"])
    for batch in batches:
        block = metrics["e2e"][f"batch_{batch}"]
        lines.append(f"| {batch} | {block['mae']:.6f} | {block['rmse']:.6f} | {block['mape']:.6f} | {block['pearson']:.6f} | {block['spearman']:.6f} |")
    (out_dir / f"comparison_summary{SUFFIX}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_markdown_row(label: str, block: dict[str, Any]) -> str:
    return f"| {label} | {block['mae']:.8f} | {block['rmse']:.8f} | {block['mape']:.8f} | {block['pearson']:.6f} | {block['spearman']:.6f} |"


def policy_key(batch: int, point: int) -> str:
    return f"batch_{batch}_point_{point:03d}"


def filter_batches(rows: list[dict[str, str]], batches: tuple[int, ...]) -> list[dict[str, str]]:
    selected = set(batches)
    return [row for row in rows if integer(row, "batch_size") in selected]


def require_count(label: str, rows: list[Any], expected: int) -> None:
    if len(rows) != expected:
        raise RuntimeError(f"{label}: expected {expected} rows, got {len(rows)}")


def required(mapping: dict[str, Any], key: str, label: str) -> Any:
    if key not in mapping:
        raise RuntimeError(f"missing {label} row for {key}")
    return mapping[key]


def number(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def integer(row: dict[str, Any], key: str) -> int:
    return int(float(row[key]))


def truthy(value: Any) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


if __name__ == "__main__":
    main()
