#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import mean
from typing import Any

from common_pareto import DEBUG_ROOT, SCENARIO, f, read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join Pareto E2E validation results for normal_02.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    e2e_path = args.output_root / "validation" / "pareto_e2e_validation.csv"
    quality_path = args.output_root / "validation" / "pareto_quality_validation.csv"
    e2e_rows = read_csv(e2e_path) if e2e_path.exists() else []
    quality_rows = read_csv(quality_path) if quality_path.exists() else []
    quality_by_point = {int(f(row, "point_index")): row for row in quality_rows}

    dense_e2e = next(
        (f(row, "e2e_total_mean_ms") for row in e2e_rows if int(f(row, "point_index")) == 0 and row.get("e2e_status") == "ok"),
        None,
    )
    dense_nll = next((f(row, "nll") for row in quality_rows if int(f(row, "point_index")) == 0), None)

    joined = []
    for row in e2e_rows:
        point = int(f(row, "point_index"))
        qrow = quality_by_point.get(point, {})
        item = dict(row)
        e2e_total = f(row, "e2e_total_mean_ms")
        item.update(
            {
                "e2e_speedup_vs_dense": dense_e2e / e2e_total if dense_e2e and e2e_total > 0 else "",
                "predicted_to_e2e_ratio": (
                    f(row, "predicted_total_latency_ms") / e2e_total if e2e_total > 0 else ""
                ),
                "nll": qrow.get("nll", ""),
                "nll_delta_vs_dense": f(qrow, "nll") - dense_nll if dense_nll is not None and qrow else "",
                "ppl": qrow.get("ppl", ""),
                "arc_acc": qrow.get("arc_acc", ""),
                "arc_acc_norm": qrow.get("arc_acc_norm", ""),
                "arc_limit": qrow.get("arc_limit", ""),
                "quality_replaced_modules": qrow.get("replaced_modules", ""),
            }
        )
        joined.append(item)

    write_csv(args.output_root / "validation" / "pareto_validation_joined.csv", joined)
    correlations = validation_correlations(joined, e2e_rows, quality_rows)
    write_csv(args.output_root / "validation" / "validation_correlations.csv", correlations)
    write_json(
        args.output_root / "validation" / "validation_summary_metadata.json",
        {
            "e2e_rows": len(e2e_rows),
            "quality_rows": len(quality_rows),
            "joined_rows": len(joined),
            "scenario": SCENARIO,
        },
    )

    write_smoke_comparison(args.output_root, joined, e2e_rows)
    print(f"wrote {len(joined)} joined validation rows")


def write_smoke_comparison(output_root: Path, joined: list[dict[str, Any]], e2e_rows: list[dict[str, Any]]) -> None:
    smoke_rows = []
    dense_row = next((r for r in e2e_rows if int(f(r, "point_index")) == 0 and r.get("e2e_status") == "ok"), None)
    dense_e2e = f(dense_row, "e2e_total_mean_ms") if dense_row else None

    for row in joined:
        e2e_mean = f(row, "e2e_total_mean_ms") if row.get("e2e_status") == "ok" else 0
        smoke_rows.append(
            {
                "row_type": "smoke_point",
                "label": f"point_{int(f(row, 'point_index')):03d}",
                "point_index": int(f(row, "point_index")),
                "quality_cost": f(row, "quality_cost"),
                "predicted_total_latency_ms": f(row, "predicted_total_latency_ms"),
                "predicted_prefill_latency_ms": f(row, "predicted_prefill_latency_ms"),
                "predicted_decode_latency_ms": f(row, "predicted_decode_latency_ms"),
                "predicted_conversion_latency_ms": f(row, "predicted_conversion_latency_ms"),
                "e2e_total_mean_ms": e2e_mean if row.get("e2e_status") == "ok" else "",
                "e2e_speedup_vs_dense": dense_e2e / e2e_mean if dense_e2e and e2e_mean > 0 else "",
                "nll": row.get("nll", ""),
                "nll_delta_vs_dense": row.get("nll_delta_vs_dense", ""),
                "arc_acc": row.get("arc_acc", ""),
                "arc_acc_norm": row.get("arc_acc_norm", ""),
                "backend_counts": str(row.get("backend_counts", "")),
                "notes": row.get("e2e_status", ""),
            }
        )

    baselines = [
        ("single_dense_bf16", "all_dense_bf16", 9101),
        ("single_dense_nvfp4", "all_dense_nvfp4", 17349),
        ("single_dense_nvfp4_prefill_marlin_decode", "all_dense_nvfp4_prefill_marlin_decode", 7762),
        ("single_marlin_nvfp4", "all_marlin_nvfp4", 7718),
        ("single_sparse_bf16", "all_sparse_bf16", 10335),
        ("single_sparse_nvfp4", "all_sparse_nvfp4", 21729),
        ("pred_policy", "pred_policy", 7282),
        ("oracle_policy", "oracle_policy", 7427),
    ]
    for label_short, label_full, e2e_ms in baselines:
        smoke_rows.append(
            {
                "row_type": "baseline",
                "label": label_full,
                "point_index": "",
                "quality_cost": "",
                "predicted_total_latency_ms": "",
                "predicted_prefill_latency_ms": "",
                "predicted_decode_latency_ms": "",
                "predicted_conversion_latency_ms": "",
                "e2e_total_mean_ms": e2e_ms,
                "e2e_speedup_vs_dense": 9101 / e2e_ms if e2e_ms > 0 else "",
                "nll": "",
                "nll_delta_vs_dense": "",
                "arc_acc": "",
                "arc_acc_norm": "",
                "backend_counts": "",
                "notes": f"from 003 sanity values",
            }
        )

    write_csv(output_root / "summary" / "normal02_smoke_comparison.csv", smoke_rows)

    write_smoke_summary_md(output_root, smoke_rows, joined, e2e_rows)


def write_smoke_summary_md(
    output_root: Path, smoke_rows: list[dict[str, Any]], joined: list[dict[str, Any]], e2e_rows: list[dict[str, Any]]
) -> None:
    ok_rows = [r for r in e2e_rows if r.get("e2e_status") == "ok"]
    replaced_ok = all(int(f(r, "replaced_linear_count")) == 224 for r in ok_rows) if ok_rows else False
    skipped_ok = all(int(f(r, "skipped_linear_count")) == 0 for r in ok_rows) if ok_rows else False

    lines = [
        "# Llama2 Normal-02 Pareto Smoke Summary",
        "",
        f"Scenario: batch_size={SCENARIO['batch_size']}, input_tokens={SCENARIO['input_tokens']}, output_tokens={SCENARIO['output_tokens']}",
        "",
        "## E2E Validation Results",
        "",
        f"- Points validated: {len(ok_rows)}",
        f"- replaced_linear_count == 224: {replaced_ok}",
        f"- skipped_linear_count == 0: {skipped_ok}",
        "",
        "| Point | Pred Total (ms) | E2E Mean (ms) | E2E Median (ms) | Speedup vs Dense | Backends |",
        "|-------|----------------|---------------|-----------------|------------------|----------|",
    ]

    for row in ok_rows:
        dense_e2e = next(
            (f(r, "e2e_total_mean_ms") for r in ok_rows if int(f(r, "point_index")) == 0), None
        )
        e2e_mean = f(row, "e2e_total_mean_ms")
        speedup = dense_e2e / e2e_mean if dense_e2e and e2e_mean > 0 else 0
        lines.append(
            f"| {int(f(row, 'point_index'))} | {f(row, 'predicted_total_latency_ms'):.1f} | "
            f"{e2e_mean:.1f} | {f(row, 'e2e_total_median_ms'):.1f} | "
            f"{speedup:.4f} | {row.get('backend_counts', '')} |"
        )

    failed_rows = [r for r in e2e_rows if r.get("e2e_status") != "ok"]
    if failed_rows:
        lines.extend(
            [
                "",
                "## Failed Smoke Points",
                "",
                "| Point | Pred Total (ms) | Backends | Reason |",
                "|-------|-----------------|----------|--------|",
            ]
        )
        for row in failed_rows:
            reason = str(row.get("unsupported_reason", "")).replace("\n", " ")
            if len(reason) > 180:
                reason = reason[:177] + "..."
            lines.append(
                f"| {int(f(row, 'point_index'))} | {f(row, 'predicted_total_latency_ms'):.1f} | "
                f"{row.get('backend_counts', '')} | {reason} |"
            )

    quality_rows = [r for r in joined if r.get("nll", "") != ""]
    if quality_rows:
        lines.extend(
            [
                "",
                "## Quality Validation Results",
                "",
                "| Point | Quality Cost | NLL | NLL Delta | ARC Acc | ARC Acc Norm |",
                "|-------|--------------|-----|-----------|---------|--------------|",
            ]
        )
        for row in quality_rows:
            lines.append(
                f"| {int(f(row, 'point_index'))} | {f(row, 'quality_cost'):.4f} | "
                f"{f(row, 'nll'):.6f} | {f(row, 'nll_delta_vs_dense'):.6f} | "
                f"{row.get('arc_acc', '')} | {row.get('arc_acc_norm', '')} |"
            )

    lines.extend(
        [
            "",
            "## Ranking Check",
            "",
        ]
    )

    pred_order = sorted(ok_rows, key=lambda r: f(r, "predicted_total_latency_ms"))
    e2e_order = sorted(ok_rows, key=lambda r: f(r, "e2e_total_mean_ms"))
    pred_ranking = {int(f(r, "point_index")): i for i, r in enumerate(pred_order)}
    e2e_ranking = {int(f(r, "point_index")): i for i, r in enumerate(e2e_order)}
    ranking_matches = pred_ranking == e2e_ranking
    lines.append(f"- Predicted ranking matches E2E ranking: {ranking_matches}")
    if not ranking_matches:
        lines.append(f"- Predicted order: {[int(f(r, 'point_index')) for r in pred_order]}")
        lines.append(f"- E2E order: {[int(f(r, 'point_index')) for r in e2e_order]}")

    lines.extend(
        [
            "",
            "## Comparison to Baselines",
            "",
            "| Label | E2E Mean (ms) | Speedup vs Dense |",
            "|-------|---------------|------------------|",
        ]
    )
    for row in smoke_rows:
        label = row["label"]
        e2e_val = row["e2e_total_mean_ms"]
        speedup = row["e2e_speedup_vs_dense"]
        lines.append(f"| {label} | {e2e_val} | {speedup} |")

    lines.extend(
        [
            "",
            "## Acceptability Check",
            "",
            f"- replaced_linear_count OK: {replaced_ok}",
            f"- skipped_linear_count OK: {skipped_ok}",
            f"- Ranking matches: {ranking_matches}",
        ]
    )

    path = output_root / "summary" / "normal02_smoke_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote smoke summary to {path}")


def validation_correlations(
    rows: list[dict[str, Any]], e2e_rows: list[dict[str, Any]], quality_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    out = []
    ok_rows = [r for r in rows if r.get("e2e_status") == "ok"]

    pairs = [(f(row, "quality_cost"), f(row, "e2e_total_mean_ms")) for row in ok_rows if row.get("e2e_total_mean_ms", "") != ""]
    if len(pairs) >= 3:
        xs = [x for x, _ in pairs]
        ys = [y for _, y in pairs]
        out.append(
            {
                "x": "quality_cost",
                "y": "e2e_total_mean_ms",
                "rows": len(pairs),
                "pearson": pearson(xs, ys),
                "spearman": spearman(xs, ys),
            }
        )

    pairs = [(f(row, "predicted_total_latency_ms"), f(row, "e2e_total_mean_ms")) for row in ok_rows if row.get("e2e_total_mean_ms", "") != ""]
    if len(pairs) >= 3:
        xs = [x for x, _ in pairs]
        ys = [y for _, y in pairs]
        out.append(
            {
                "x": "predicted_total_latency_ms",
                "y": "e2e_total_mean_ms",
                "rows": len(pairs),
                "pearson": pearson(xs, ys),
                "spearman": spearman(xs, ys),
            }
        )

    pairs = [(f(row, "predicted_total_latency_ms"), f(row, "e2e_prefill_mean_ms")) for row in ok_rows if row.get("e2e_prefill_mean_ms", "") != ""]
    if len(pairs) >= 3:
        xs = [x for x, _ in pairs]
        ys = [y for _, y in pairs]
        out.append(
            {
                "x": "predicted_total_latency_ms",
                "y": "e2e_prefill_mean_ms",
                "rows": len(pairs),
                "pearson": pearson(xs, ys),
                "spearman": spearman(xs, ys),
            }
        )

    for metric in ("nll", "arc_acc_norm"):
        pairs = [(f(row, "quality_cost"), f(row, metric)) for row in quality_rows if row.get(metric, "") != ""]
        if len(pairs) < 3:
            continue
        xs = [x for x, _ in pairs]
        ys = [y for _, y in pairs]
        out.append(
            {
                "x": "quality_cost",
                "y": metric,
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


if __name__ == "__main__":
    main()
