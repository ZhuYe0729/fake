#!/usr/bin/env python3
"""Summarize phase-switch speed benchmark JSON files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    rows = []
    for path in sorted(result_dir.glob("*.json")):
        if path.name == Path(args.output_json).name or path.name.endswith(
                "_trace.json"):
            continue
        with path.open() as f:
            row = json.load(f)
        if "generate_s" in row:
            rows.append(row)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scenario"]].append(row)

    summary = []
    for scenario, items in sorted(grouped.items()):
        best = min(items, key=lambda x: x["generate_s"])
        dense = next((x for x in items if x["label"] == "dense_nvfp4"), None)
        w4 = next((x for x in items if x["label"] == "w4a16_ours"), None)
        phase = next((x for x in items if x["label"] == "phase_switch"), None)
        entry = {
            "scenario": scenario,
            "best_label": best["label"],
            "rows": sorted(items, key=lambda x: x["label"]),
        }
        if phase and dense:
            entry["phase_vs_dense_speedup"] = (
                dense["generate_s"] / phase["generate_s"])
        if phase and w4:
            entry["phase_vs_w4a16_speedup"] = (
                w4["generate_s"] / phase["generate_s"])
        summary.append(entry)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    lines = ["# Phase Switch Speed Benchmark", ""]
    for entry in summary:
        lines.append(f"## {entry['scenario']}")
        lines.append("")
        lines.append(
            "| label | generate_s | output tok/s | total tok/s | init_s |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in entry["rows"]:
            lines.append("| {label} | {generate_s:.4f} | "
                         "{output_tokens_per_s:.2f} | {total_tokens_per_s:.2f} "
                         "| {init_s:.2f} |".format(**row))
        lines.append("")
        lines.append(f"best: `{entry['best_label']}`")
        if "phase_vs_dense_speedup" in entry:
            lines.append("phase vs dense_nvfp4 speedup: "
                         f"{entry['phase_vs_dense_speedup']:.3f}x")
        if "phase_vs_w4a16_speedup" in entry:
            lines.append("phase vs w4a16_ours speedup: "
                         f"{entry['phase_vs_w4a16_speedup']:.3f}x")
        lines.append("")

    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
