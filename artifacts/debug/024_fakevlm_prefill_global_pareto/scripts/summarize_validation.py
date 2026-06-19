#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from common_fakevlm_pareto import DEBUG_ROOT, f, read_csv, source_020_accuracy, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize FakeVLM Pareto validation results.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = read_csv(args.output_root / "validation" / "selected_pareto_points.csv")
    speed = read_optional(args.output_root / "validation" / "pareto_speed_validation.csv")
    quality = read_optional(args.output_root / "quality" / "validation_quality.csv")
    joined = join_rows(selected, speed, quality)
    write_csv(args.output_root / "validation" / "pareto_validation_joined.csv", joined)
    write_summary(args.output_root, joined)
    plot(args.output_root, joined)
    print(f"wrote joined rows={len(joined)}")


def read_optional(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.exists() else []


def join_rows(selected: list[dict[str, str]], speed: list[dict[str, str]], quality: list[dict[str, str]]) -> list[dict[str, str]]:
    speed_map = {(int(float(row["batch_size"])), int(float(row["point_index"]))): row for row in speed}
    quality_map = {(int(float(row["batch_size_for_policy"])), int(float(row["point_index"]))): row for row in quality if row.get("batch_size_for_policy") not in {"", None}}
    rows = []
    for row in selected:
        key = (int(float(row["batch_size"])), int(float(row["point_index"])))
        item = dict(row)
        srow = speed_map.get(key, {})
        qrow = quality_map.get(key, {})
        for field in ("e2e_prefill_mean_ms", "e2e_prefill_p50_ms", "samples_per_sec"):
            item[field] = srow.get(field, "")
        item["global_accuracy"] = qrow.get("global_accuracy", "")
        rows.append(item)
    return rows


def write_summary(output_root: Path, rows: list[dict[str, str]]) -> None:
    lines = ["# FakeVLM Prefill Global Pareto Summary", ""]
    uniform = source_020_accuracy()
    if uniform:
        lines.append("## Uniform Accuracy Baselines")
        lines.append("")
        lines.append("| Method | Accuracy |")
        lines.append("|---|---:|")
        for row in uniform:
            lines.append(f"| `{row['method']}` | {row['global_accuracy']} |")
        lines.append("")
    lines.append("## Validated Pareto Points")
    lines.append("")
    lines.append("| Batch | Point | Pred latency ms | Pred quality cost | E2E ms | Accuracy | Counts |")
    lines.append("|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        counts = ", ".join(f"{k.removeprefix('count_')}={row[k]}" for k in row if k.startswith("count_") and row[k] not in {"0", "0.0", ""})
        lines.append(
            f"| {row['batch_size']} | {row['point_index']} | {f(row, 'latency_ms'):.3f} | "
            f"{f(row, 'quality_cost'):.6g} | {row.get('e2e_prefill_mean_ms', '')} | {row.get('global_accuracy', '')} | {counts} |"
        )
    (output_root / "summary").mkdir(parents=True, exist_ok=True)
    (output_root / "summary" / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(output_root: Path, rows: list[dict[str, str]]) -> None:
    complete = [row for row in rows if row.get("e2e_prefill_mean_ms") and row.get("global_accuracy")]
    if not complete:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for batch in sorted({int(float(row["batch_size"])) for row in complete}):
        subset = [row for row in complete if int(float(row["batch_size"])) == batch]
        ax.scatter([f(row, "e2e_prefill_mean_ms") for row in subset], [f(row, "global_accuracy") for row in subset], label=f"batch {batch}")
    ax.set_xlabel("Measured prefill latency (ms)")
    ax.set_ylabel("FakeClue global accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    (output_root / "plots").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_root / "plots" / "speed_vs_accuracy.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
