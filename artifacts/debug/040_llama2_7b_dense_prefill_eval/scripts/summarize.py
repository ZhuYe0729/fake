#!/usr/bin/env python3
"""Build a compact metric and timing table from completed task results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("wikitext", "c4", "winogrande", "arc_easy", "mmlu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="full")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for task in TASKS:
        path = ROOT / "tasks" / task / args.profile / "result.json"
        if not path.exists():
            rows.append({"task": task, "status": "missing", "result_path": str(path)})
            continue
        payload = json.loads(path.read_text())
        metrics = payload.get("metrics", {})
        row = {
            "task": task,
            "status": "ok",
            "elapsed_seconds": payload.get("elapsed_seconds"),
            "elapsed_minutes": payload.get("elapsed_minutes"),
            "num_fewshot": payload.get("num_fewshot"),
            "limit": payload.get("limit"),
            "result_path": str(path),
        }
        row.update(metrics)
        rows.append(row)

    run_path = ROOT / "runs" / args.profile / "run_summary.json"
    run = json.loads(run_path.read_text()) if run_path.exists() else {}
    summary = ROOT / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (summary / f"results_{args.profile}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    wall_clock = f"{run['elapsed_seconds']} s" if "elapsed_seconds" in run else "not tracked (tasks started separately)"
    lines = [f"# Dense Llama2 quality results ({args.profile})", "", f"Parallel wall-clock: {wall_clock}", "", "| task | status | primary metrics | total minutes |", "|---|---|---|---:|"]
    for row in rows:
        metrics = ", ".join(f"{key}={value}" for key, value in row.items() if key.startswith(("acc,", "acc_norm,", "word_perplexity,", "byte_perplexity,", "bits_per_byte,"))) or "—"
        minutes = row.get("elapsed_minutes", "—")
        lines.append(f"| {row['task']} | {row['status']} | {metrics} | {minutes} |")
    (summary / f"results_{args.profile}.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
