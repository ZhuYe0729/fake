#!/usr/bin/env python3
"""Summarize max-speed predictor, vLLM speed, and PMPD quality outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    rows = []
    for scenario_dir in sorted((args.root / "max_speed").glob("*")):
        metadata = read_json(scenario_dir / "policy/policy_metadata.json")
        speed = official_speed(scenario_dir)
        if metadata:
            rows.append({"scenario": scenario_dir.name, "predicted_linear_latency_ms": metadata.get("predicted_linear_latency_ms"), "module_count": metadata.get("module_count"), **speed})
    output = args.root / "max_speed/summary"; output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "speed_summary.csv", rows)
    quality = []
    for metrics in sorted((args.root / "max_speed").glob("*/results/quality/*/metrics.json")):
        payload = read_json(metrics)
        quality.append({"scenario": metrics.parents[3].name, "dataset": metrics.parent.name, "rougeL_percent": payload.get("rougeL_percent"), "bert_score_percent": payload.get("bert_score_percent"), "sacre_bleu": payload.get("sacre_bleu"), "metrics_path": str(metrics)})
    write_csv(output / "quality_summary.csv", quality)
    (output / "summary.md").write_text("# Llama2-7B-Chat Ours Max-Speed Summary\n\nPareto search: TODO.\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def first_csv(path: Path) -> dict:
    if not path.exists(): return {}
    with path.open(newline="", encoding="utf-8") as handle: return next(csv.DictReader(handle), {})


def official_speed(scenario_dir: Path) -> dict:
    if scenario_dir.name == "prefill_only":
        summary = read_json(scenario_dir / "baseline_aligned_speed" / "summary.json")
        if summary:
            return {"speed_protocol": "baseline_aligned", "e2e_mean_ms": summary.get("mean_ms"), "e2e_median_ms": summary.get("median_ms"), "ttft_median_ms": summary.get("median_ms"), "tpot_ms": 0.0, "repeats": summary.get("repeats")}
    summaries = read_json(scenario_dir.parent / "summary" / "fresh_process_speed_summary.json")
    if isinstance(summaries, list):
        for summary in summaries:
            if summary.get("scenario") == scenario_dir.name:
                return {"speed_protocol": "phase_fresh_process", **summary}
    return first_csv(scenario_dir / "results/speed/summary.csv")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
