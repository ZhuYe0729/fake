#!/usr/bin/env python3
"""Join actual vLLM runtime quality with existing measured speed."""
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
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--profile", default="full")
    a = parser.parse_args(); manifest = json.loads((DEBUG / "manifest/policies.json").read_text()); rows = []
    for item in manifest["policies"]:
        row = {key: item.get(key, "") for key in ("label", "family", "recommended_use", "e2e_ms", "speedup", "speed_source")}
        row["policy"] = row.pop("label"); complete = True
        for task, specs in METRICS.items():
            try: payload = json.loads((DEBUG / "results" / item["label"] / task / a.profile / "result.json").read_text())
            except (OSError, ValueError): payload = {}
            values = payload.get("metrics", {}); complete &= bool(values)
            for source, target in specs: row[target] = values.get(source, "")
        row["status"] = "complete" if complete else "pending"; rows.append(row)
    out = DEBUG / "summary"; out.mkdir(parents=True, exist_ok=True)
    columns = ["family", "policy", "recommended_use", "e2e_ms", "speedup", "wikitext_word_ppl", "winogrande_acc", "arc_easy_acc", "arc_easy_acc_norm", "arc_challenge_acc", "arc_challenge_acc_norm", "mmlu_acc", "status", "speed_source"]
    with (out / f"{a.profile}_runtime_quality.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    lines = ["# Llama2 prefill-only: actual vLLM runtime quality", "", "| " + " | ".join(columns[:-1]) + " |", "|" + "|".join(["---"] * (len(columns) - 1)) + "|"]
    for row in rows:
        values = []
        for key in columns[:-1]:
            value = row.get(key, "")
            values.append(f"{value:.6g}" if isinstance(value, float) else str(value or "—"))
        lines.append("| " + " | ".join(values) + " |")
    (out / f"{a.profile}_runtime_quality.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
