#!/usr/bin/env python3
"""Join Llama3.1 actual vLLM runtime quality with measured speed."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DEBUG = Path(__file__).resolve().parents[1]
METRICS = {
    "wikitext": (("word_perplexity,none", "wikitext_word_ppl"),),
    "winogrande": (("acc,none", "winogrande_acc"),),
    "arc_easy": (("acc,none", "arc_easy_acc"), ("acc_norm,none", "arc_easy_acc_norm")),
    "arc_challenge": (("acc,none", "arc_challenge_acc"), ("acc_norm,none", "arc_challenge_acc_norm")),
    "mmlu": (("acc,none", "mmlu_acc"),),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="full")
    args = parser.parse_args()
    manifest = json.loads((DEBUG / "manifest/policies.json").read_text())
    rows = []
    for item in manifest["policies"]:
        row = {key: item.get(key, "") for key in ("family", "recommended_use", "e2e_ms", "speedup", "speed_source")}
        row["policy"] = item["label"]
        complete = True
        for task, specs in METRICS.items():
            try:
                metrics = json.loads((DEBUG / "results" / item["label"] / task / args.profile / "result.json").read_text())["metrics"]
            except (OSError, ValueError, KeyError):
                metrics = {}
            complete &= bool(metrics)
            for source, target in specs:
                row[target] = metrics.get(source, "")
        row["status"] = "complete" if complete else "pending"
        rows.append(row)
    columns = ["family", "policy", "recommended_use", "e2e_ms", "speedup", "wikitext_word_ppl", "winogrande_acc", "arc_easy_acc", "arc_easy_acc_norm", "arc_challenge_acc", "arc_challenge_acc_norm", "mmlu_acc", "status", "speed_source"]
    output = DEBUG / "summary"
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{args.profile}_runtime_quality.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Llama3.1 prefill-only: actual vLLM runtime quality", "", "| " + " | ".join(columns[:-1]) + " |", "|" + "|".join(["---"] * (len(columns) - 1)) + "|"]
    for row in rows:
        values = [f"{value:.6g}" if isinstance(value := row.get(key, ""), float) else str(value or "—") for key in columns[:-1]]
        lines.append("| " + " | ".join(values) + " |")
    (output / f"{args.profile}_runtime_quality.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
