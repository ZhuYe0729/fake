#!/usr/bin/env python3
"""Summarize speed and PMPD quality outputs for llama2-7b-chat baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=BASELINE_ROOT / "results/summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    speed_rows = read_speed(args.root / "results/speed/summary.csv")
    quality_rows = read_quality(args.root / "results/quality")
    write_csv(args.output_dir / "speed_summary.csv", speed_rows)
    write_csv(args.output_dir / "quality_summary.csv", quality_rows)
    write_markdown(args.output_dir / "baseline_summary.md", speed_rows, quality_rows)


def read_speed(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_quality(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(root.glob("*/*/metrics.json")):
        payload = json.loads(metrics_path.read_text())
        method = metrics_path.parents[1].name
        dataset = metrics_path.parent.name
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "num_samples": payload.get("num_samples"),
                "empty_predictions": payload.get("empty_predictions"),
                "rougeL_percent": payload.get("rougeL_percent"),
                "bert_score_percent": payload.get("bert_score_percent"),
                "sacre_bleu": payload.get("sacre_bleu"),
                "tokens_per_second": payload.get("tokens_per_second"),
                "metrics_path": str(metrics_path),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not keys:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, speed_rows: list[dict[str, Any]], quality_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Llama2-7B-Chat Baseline Summary",
        "",
        "CNN/DM quality uses `cnn_dm_1000`, the fixed 1000-example subset.",
        "IWSLT uses the Llama-2-chat tokenizer for length filtering because the PMPD Vicuna tokenizer is unavailable locally; treat it as a non-strict PMPD result.",
        "",
        "## Speed",
        "",
    ]
    if speed_rows:
        lines.append("| method | scenario | e2e median ms | TTFT ms | TPOT ms | total tok/s | status |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
        for row in speed_rows:
            lines.append(
                "| {method} | {scenario} | {e2e} | {ttft} | {tpot} | {tok} | {status} |".format(
                    method=row.get("method", ""),
                    scenario=row.get("scenario", ""),
                    e2e=fmt(row.get("e2e_median_ms")),
                    ttft=fmt(row.get("ttft_median_ms")),
                    tpot=fmt(row.get("tpot_ms")),
                    tok=fmt(row.get("total_tokens_per_s")),
                    status=row.get("status", ""),
                )
            )
    else:
        lines.append("_No speed results found._")

    lines.extend(["", "## Quality", ""])
    if quality_rows:
        lines.append("| method | dataset | samples | empty | Rouge-L | BERTScore | SacreBLEU |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for row in quality_rows:
            lines.append(
                "| {method} | {dataset} | {samples} | {empty} | {rouge} | {bert} | {bleu} |".format(
                    method=row.get("method", ""),
                    dataset=row.get("dataset", ""),
                    samples=row.get("num_samples", ""),
                    empty=row.get("empty_predictions", ""),
                    rouge=fmt(row.get("rougeL_percent")),
                    bert=fmt(row.get("bert_score_percent")),
                    bleu=fmt(row.get("sacre_bleu")),
                )
            )
    else:
        lines.append("_No quality results found._")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
