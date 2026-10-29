#!/usr/bin/env python3
"""Summarize phase-switch overhead debug variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = [
    "phase_baseline",
    "phase_no_trace",
    "phase_no_decode_gc",
    "phase_wait_once",
    "phase_combined",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    rows = []
    for label in LABELS:
        base = f"{label}_{args.scenario}"
        prefill = load_json(result_dir / f"{base}_o1.json")
        full = load_json(result_dir / f"{base}_o{args.output_len}.json")
        prefill_s = prefill["generate_s"]
        total_s = full["generate_s"]
        decode_s = max(total_s - prefill_s, 0.0)
        decode_tokens = max(full["output_tokens"] - prefill["output_tokens"],
                            0)
        rows.append({
            "label": label,
            "prefill_s_approx": prefill_s,
            "decode_s_approx": decode_s,
            "total_s": total_s,
            "decode_tokens_per_s": decode_tokens / decode_s
            if decode_s > 0 else None,
            "total_tokens_per_s": full["total_tokens_per_s"],
        })
    baseline = rows[0]
    for row in rows:
        row["total_delta_vs_baseline_s"] = row["total_s"] - baseline[
            "total_s"]
        row["decode_delta_vs_baseline_s"] = row["decode_s_approx"] - baseline[
            "decode_s_approx"]

    out = {"scenario": args.scenario, "rows": rows}
    Path(args.output_json).write_text(json.dumps(out, indent=2) + "\n")

    lines = [
        "# Phase Switch Overhead Debug",
        "",
        f"scenario: `{args.scenario}`",
        "",
        "| label | prefill_s | decode_s | total_s | decode delta | total delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {prefill_s_approx:.4f} | {decode_s_approx:.4f} | "
            "{total_s:.4f} | {decode_delta_vs_baseline_s:.4f} | "
            "{total_delta_vs_baseline_s:.4f} |".format(**row))
    Path(args.output_md).write_text("\n".join(lines) + "\n",
                                    encoding="utf-8")


if __name__ == "__main__":
    main()
