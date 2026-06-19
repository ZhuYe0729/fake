#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1]
HYBRID_FAMILIES = ("manual_profile", "latency_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize FakeVLM prefill-only hybrid speed results.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    speed_path = args.output_root / "speed" / "prefill_speed.csv"
    rows = read_csv(speed_path)
    summary = summarize(rows)
    write_csv(args.output_root / "summary" / "prefill_speed_summary.csv", summary)
    write_markdown(args.output_root / "summary" / "prefill_speed_summary.md", summary)
    print(f"wrote {len(summary)} summary rows to {args.output_root / 'summary'}")


def summarize(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_batch: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        try:
            batch = int(row["batch_size"])
        except (KeyError, ValueError):
            continue
        by_batch.setdefault(batch, []).append(row)

    output = []
    for batch in sorted(by_batch):
        batch_rows = latest_by_family(by_batch[batch])
        uniform = [row for family, row in batch_rows.items() if family.startswith("uniform_")]
        best_uniform = best_by_samples_per_sec(uniform)
        best_uniform_sps = to_float(best_uniform.get("samples_per_sec")) if best_uniform else math.nan
        for family in HYBRID_FAMILIES:
            row = batch_rows.get(family)
            if row is None:
                continue
            sps = to_float(row.get("samples_per_sec"))
            output.append(
                {
                    "batch_size": batch,
                    "family": family,
                    "latency_mean_ms": row.get("latency_mean_ms", ""),
                    "samples_per_sec": row.get("samples_per_sec", ""),
                    "best_uniform_family": best_uniform.get("family", "") if best_uniform else "",
                    "best_uniform_latency_mean_ms": best_uniform.get("latency_mean_ms", "") if best_uniform else "",
                    "best_uniform_samples_per_sec": best_uniform.get("samples_per_sec", "") if best_uniform else "",
                    "speedup_vs_best_uniform": (
                        "" if math.isnan(sps) or math.isnan(best_uniform_sps) or best_uniform_sps <= 0 else f"{sps / best_uniform_sps:.6f}"
                    ),
                    "replaced_linear_count": row.get("replaced_linear_count", ""),
                    "skipped_linear_count": row.get("skipped_linear_count", ""),
                    "backend_counts": row.get("backend_counts", ""),
                    "device_name": row.get("device_name", ""),
                    "policy_path": row.get("policy_path", ""),
                }
            )
    return output


def latest_by_family(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    latest: dict[str, dict[str, str]] = {}
    for row in rows:
        family = row.get("family", "")
        if not family:
            continue
        old = latest.get(family)
        if old is None or row.get("timestamp", "") >= old.get("timestamp", ""):
            latest[family] = row
    return latest


def best_by_samples_per_sec(rows: list[dict[str, str]]) -> dict[str, str]:
    best: dict[str, str] = {}
    best_sps = -1.0
    for row in rows:
        sps = to_float(row.get("samples_per_sec"))
        if not math.isnan(sps) and sps > best_sps:
            best = row
            best_sps = sps
    return best


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FakeVLM Prefill Speed Summary",
        "",
        "| Batch | Family | Latency ms | Samples/s | Best uniform | Speedup |",
        "|---:|---|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['batch_size']} | {row['family']} | {row['latency_mean_ms']} | "
            f"{row['samples_per_sec']} | {row['best_uniform_family']} | {row['speedup_vs_best_uniform']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


if __name__ == "__main__":
    main()
