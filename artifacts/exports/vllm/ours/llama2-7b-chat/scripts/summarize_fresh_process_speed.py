#!/usr/bin/env python3
"""Summarize established fresh-process phase-heterogeneous speed runs."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); args = parser.parse_args()
    rows = []
    for scenario in ("prefill_only", "prefill_decode"):
        runs = args.root / "max_speed" / scenario / "fresh_process_speed" / "runs"
        o1 = values(runs, "measured_o1_r*.json")
        output_len = 1 if scenario == "prefill_only" else 80
        on = values(runs, f"measured_o{output_len}_r*.json")
        if not o1 or not on: continue
        row = {"scenario": scenario, "repeats": len(on), "ttft_median_ms": 1000 * statistics.median(o1), "e2e_median_ms": 1000 * statistics.median(on), "e2e_mean_ms": 1000 * statistics.mean(on)}
        row["tpot_ms"] = 0.0 if output_len == 1 else 1000 * (statistics.median(on) - statistics.median(o1)) / (output_len - 1)
        rows.append(row)
    out = args.root / "max_speed" / "summary"; out.mkdir(parents=True, exist_ok=True)
    (out / "fresh_process_speed_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))

def values(root: Path, pattern: str) -> list[float]:
    return [float(json.loads(path.read_text())["generate_s"]) for path in sorted(root.glob(pattern))]

if __name__ == "__main__": main()
