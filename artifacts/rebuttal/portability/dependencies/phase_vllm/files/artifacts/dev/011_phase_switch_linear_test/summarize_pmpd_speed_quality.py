#!/usr/bin/env python3
"""Merge PMPD quality results with previous speed results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LABEL_MAP = {
    "dense_bf16": "bf16",
    "dense_nvfp4": "dense_nvfp4",
    "w4a16": "w4a16_ours",
    "dense_nvfp4_plus_w4a16": "phase_switch_notrace",
}

DISPLAY = {
    "dense_bf16": "dense BF16",
    "dense_nvfp4": "dense NVFP4",
    "w4a16": "W4A16 Marlin",
    "dense_nvfp4_plus_w4a16": "dense NVFP4 + W4A16",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pmpd-dir", required=True)
    parser.add_argument("--speed-summary", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    pmpd_dir = Path(args.pmpd_dir)
    speed = load(Path(args.speed_summary))
    speed_rows = {row["label"]: row for row in speed["rows"]}
    rows = []
    for model_id, speed_label in LABEL_MAP.items():
        metrics = load(pmpd_dir / f"{model_id}_metrics.json")
        srow = speed_rows[speed_label]
        rows.append({
            "model": model_id,
            "display": DISPLAY[model_id],
            "rougeL_percent": metrics["rougeL_percent"],
            "bert_score_percent": metrics["bert_score_percent"],
            "empty_predictions": metrics["empty_predictions"],
            "num_samples": metrics["num_samples"],
            "prefill_s": srow["prefill_s"],
            "decode_s": srow["decode_s"],
            "total_s": srow["total_s"],
            "speedup_vs_bf16": srow["speedup_vs_bf16"],
        })
    summary = {
        "quality_dataset": "cnn_dm_1000 first 100 examples",
        "speed_scenario": speed["scenario"],
        "rows": rows,
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# PMPD Quality + Speed Summary",
        "",
        "Quality: `cnn_dm_1000` first 100 examples, PMPD-style prompt/metrics.",
        f"Speed: `{speed['scenario']}`.",
        "",
        "| Method | Rouge-L | BERTScore | Empty | Prefill | Decoding | Total | Speedup vs BF16 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['display']} | {row['rougeL_percent']:.2f} | "
            f"{row['bert_score_percent']:.2f} | "
            f"{row['empty_predictions']}/{row['num_samples']} | "
            f"{row['prefill_s']:.4f}s | {row['decode_s']:.4f}s | "
            f"{row['total_s']:.4f}s | {row['speedup_vs_bf16']:.2f}x |")
    lines.append("")
    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
