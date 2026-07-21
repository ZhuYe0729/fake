#!/usr/bin/env python3
"""Join multi-task quality results with existing measured prefill-only speed."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEBUG = Path(__file__).resolve().parents[1]
MANIFEST = DEBUG / "manifest/policies.json"
TASK_METRICS = {
    "wikitext": (("word_perplexity,none", "wikitext_word_ppl"), ("byte_perplexity,none", "wikitext_byte_ppl"), ("bits_per_byte,none", "wikitext_bits_per_byte")),
    "winogrande": (("acc,none", "winogrande_acc"),),
    "arc_easy": (("acc,none", "arc_easy_acc"), ("acc_norm,none", "arc_easy_acc_norm")),
    "mmlu": (("acc,none", "mmlu_acc"),),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="full")
    return parser.parse_args()


def read_result(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def markdown(rows: list[dict[str, Any]]) -> str:
    columns = ["family", "policy", "recommended_use", "e2e_ms", "speedup", "wikitext_word_ppl", "winogrande_acc", "arc_easy_acc", "arc_easy_acc_norm", "mmlu_acc", "status"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            values.append(f"{value:.6g}" if isinstance(value, float) else str(value or "—"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    manifest = json.loads(MANIFEST.read_text())
    all_rows: list[dict[str, Any]] = []
    for item in manifest["policies"]:
        row = {key: item.get(key, "") for key in ("model", "family", "label", "recommended_use", "e2e_ms", "speedup", "speed_source")}
        row["policy"] = row.pop("label")
        complete = True
        for task, metric_names in TASK_METRICS.items():
            payload = read_result(DEBUG / "results" / item["model"] / item["label"] / task / args.profile / "result.json")
            metrics = payload.get("metrics", {})
            if not metrics:
                complete = False
            for metric, key in metric_names:
                row[key] = metrics.get(metric, "")
        row["status"] = "complete" if complete else "pending"
        all_rows.append(row)
    summary = DEBUG / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in all_rows for key in row})
    with (summary / f"{args.profile}_all_results_long.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)
    for model in manifest["models"]:
        rows = [row for row in all_rows if row["model"] == model]
        with (summary / f"{args.profile}_{model}_prefill_only_multitask.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        (summary / f"{args.profile}_{model}_prefill_only_multitask.md").write_text(
            f"# {model}: prefill-only multi-task quality ({args.profile})\n\n" + markdown(rows)
        )
    print(f"wrote summaries for {len(all_rows)} policies under {summary}")


if __name__ == "__main__":
    main()
