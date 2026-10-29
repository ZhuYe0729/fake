#!/usr/bin/env python3
"""Summarize no-trace end-to-end decomposed benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = ["bf16", "dense_nvfp4", "w4a16_ours", "phase_switch_notrace"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    rows = []
    for label in LABELS:
        base = f"{label}_{args.scenario}"
        o1 = load(result_dir / f"{base}_o1.json")
        full = load(result_dir / f"{base}_o{args.output_len}.json")
        prefill_s = o1["generate_s"]
        total_s = full["generate_s"]
        decode_s = max(total_s - prefill_s, 0.0)
        rows.append({
            "label": label,
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "total_s": total_s,
            "total_tokens_per_s": full["total_tokens_per_s"],
        })
    bf16_total = rows[0]["total_s"]
    for row in rows:
        row["speedup_vs_bf16"] = bf16_total / row["total_s"]

    summary = {
        "scenario": args.scenario,
        "method": "prefill approximated by output_len=1; decode is output_len=N minus output_len=1",
        "rows": rows,
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# No-Trace End-to-End Benchmark",
        "",
        f"scenario: `{args.scenario}`",
        "",
        "| Method | Prefill | Decoding | Total | Total speedup vs BF16 |",
        "|---|---:|---:|---:|---:|",
    ]
    names = {
        "bf16": "dense BF16",
        "dense_nvfp4": "dense NVFP4",
        "w4a16_ours": "W4A16 Marlin",
        "phase_switch_notrace": "dense NVFP4 + W4A16",
    }
    for row in rows:
        lines.append(
            f"| {names[row['label']]} | {row['prefill_s']:.4f}s | "
            f"{row['decode_s']:.4f}s | {row['total_s']:.4f}s | "
            f"{row['speedup_vs_bf16']:.2f}x |")
    lines.append("")
    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
