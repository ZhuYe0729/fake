#!/usr/bin/env python3
"""Summarize Llama3.1 ours max-speed outputs and comparable uniform baselines."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}

def rows_from_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f)) if path.exists() else []

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)

def fresh_speed(root: Path) -> dict:
    runs = root / "max_speed/prefill_decode/fresh_process_speed/runs"
    def values(pattern: str) -> list[float]: return [float(read_json(x)["generate_s"]) for x in sorted(runs.glob(pattern))]
    o1, o80 = values("measured_o1_r*.json"), values("measured_o80_r*.json")
    if not o1 or not o80:
        rows = rows_from_csv(root / "max_speed/prefill_decode/results/speed/summary.csv")
        return rows[0] if rows else {}
    return {"speed_protocol": "phase_fresh_process", "repeats": len(o80), "ttft_median_ms": 1000 * statistics.median(o1), "e2e_median_ms": 1000 * statistics.median(o80), "e2e_mean_ms": 1000 * statistics.mean(o80), "tpot_ms": 1000 * (statistics.median(o80) - statistics.median(o1)) / 79}

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1]); args = p.parse_args()
    root = args.root; summary = root / "max_speed/summary"; summary.mkdir(parents=True, exist_ok=True)
    ours = []
    for scenario in ("prefill_only", "prefill_decode"):
        meta = read_json(root / f"max_speed/{scenario}/policy/policy_metadata.json")
        if not meta: continue
        speed = read_json(root / "max_speed/prefill_only/baseline_aligned_speed/summary.json") if scenario == "prefill_only" else fresh_speed(root)
        if scenario == "prefill_only" and speed:
            speed = {"speed_protocol": "baseline_aligned", "e2e_mean_ms": speed["mean_ms"], "e2e_median_ms": speed["median_ms"], "ttft_median_ms": speed["median_ms"], "tpot_ms": 0.0, "repeats": speed["repeats"]}
        ours.append({"method": "ours_max_speed", "scenario": scenario, "predicted_linear_latency_ms": meta["predicted_linear_latency_ms"], "module_count": meta["module_count"], **speed})
    write_csv(summary / "speed_summary.csv", ours)
    quality = []
    for metric in sorted((root / "max_speed").glob("*/results/quality/*/metrics.json")):
        data = read_json(metric)
        quality.append({"method": "ours_max_speed", "scenario": metric.parents[3].name, "dataset": metric.parent.name, "rougeL_percent": data.get("rougeL_percent"), "bert_score_percent": data.get("bert_score_percent"), "sacre_bleu": data.get("sacre_bleu"), "metrics_path": str(metric)})
    write_csv(summary / "quality_summary.csv", quality)
    baseline = root.parents[1] / "baselines/llama3.1-8b-instruct/results/summary"
    baseline_speed = rows_from_csv(baseline / "speed_summary.csv")
    comparison = []
    for ours_row in ours:
        same = [x for x in baseline_speed if x.get("scenario") == ours_row["scenario"]]
        dense = next((x for x in same if x.get("method") == "dense_bf16"), None)
        best = min(same, key=lambda x: float(x["e2e_median_ms"])) if same else None
        ours_ms = float(ours_row["e2e_median_ms"]) if ours_row.get("e2e_median_ms") else None
        comparison.append({**ours_row, "speedup_vs_dense": float(dense["e2e_median_ms"]) / ours_ms if dense and ours_ms else None, "speedup_vs_best_uniform": float(best["e2e_median_ms"]) / ours_ms if best and ours_ms else None, "best_uniform_method": best.get("method") if best else None})
    write_csv(summary / "comparison_summary.csv", comparison)
    (summary / "summary.md").write_text("# Llama-3.1-8B-Instruct Ours Max-Speed Summary\n\n`ours_max_speed` is unconstrained by task quality; use the quality table to assess its trade-off.\n", encoding="utf-8")

if __name__ == "__main__":
    main()
