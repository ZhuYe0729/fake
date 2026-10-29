#!/usr/bin/env python3
"""Merge a decomposed summary with separately measured BF16 results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-summary", required=True)
    parser.add_argument("--bf16-o1", required=True)
    parser.add_argument("--bf16-ofull", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def load(path: str) -> dict:
    with Path(path).open() as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    summary = load(args.base_summary)
    bf16_o1 = load(args.bf16_o1)
    bf16_full = load(args.bf16_ofull)
    prefill_s = bf16_o1["generate_s"]
    total_s = bf16_full["generate_s"]
    decode_s = max(total_s - prefill_s, 0.0)
    decode_tokens = max(bf16_full["output_tokens"] -
                        bf16_o1["output_tokens"], 0)
    bf16_row = {
        "label": "bf16",
        "prefill_s_approx": prefill_s,
        "decode_s_approx": decode_s,
        "total_s": total_s,
        "prefill_tokens_per_s": bf16_full["prompt_tokens"] / prefill_s,
        "decode_tokens_per_s": decode_tokens / decode_s
        if decode_s > 0 else None,
        "total_tokens_per_s": bf16_full["total_tokens_per_s"],
        "prompt_tokens": bf16_full["prompt_tokens"],
        "decode_tokens_for_decode_phase": decode_tokens,
    }
    rows = [bf16_row] + summary["rows"]
    merged = dict(summary)
    merged["rows"] = rows
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(merged, indent=2) + "\n")

    lines = [
        "# Decomposed Phase Switch Benchmark With BF16",
        "",
        f"scenario: `{summary['scenario']}`",
        "",
        "Prefill is approximated by the same prompt batch with `output_len=1`; "
        "decode is `output_len=N` total time minus the `output_len=1` run.",
        "",
        "| label | prefill_s | decode_s | total_s | prefill tok/s | decode tok/s | total tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        decode_tps = row["decode_tokens_per_s"]
        lines.append(
            "| {label} | {prefill_s_approx:.4f} | {decode_s_approx:.4f} | "
            "{total_s:.4f} | {prefill_tokens_per_s:.2f} | "
            "{decode_tps} | {total_tokens_per_s:.2f} |".format(
                decode_tps=f"{decode_tps:.2f}" if decode_tps else "n/a",
                **row))
    overhead = summary["switch_overhead"]
    lines.extend([
        "",
        "## Switch overhead",
        "",
        f"- non-overlap switch copy: {overhead['non_overlap_switch_s']:.4f}s",
        f"- exposed boundary wait with overlap: {overhead['overlap_boundary_wait_s']:.4f}s",
        f"- hidden by overlap: {overhead['hidden_switch_s']:.4f}s",
        "- sync total minus overlap total: "
        f"{overhead['sync_total_minus_overlap_total_s']:.4f}s",
        "",
    ])
    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
