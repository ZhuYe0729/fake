#!/usr/bin/env python3
"""Render compact one-policy-per-row source summaries for the consolidated bundle."""
from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/debug/060_two_model_two_scenario_result_consolidation"


def key(policy: str) -> tuple[int, int, str]:
    match = re.search(r"(\d+)$", policy)
    return (0, int(match.group(1)), policy) if match else (1, 0, policy)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def is_uniform(model: str, scenario: str, row: dict[str, str]) -> bool:
    if row.get("family") == "uniform":
        return True
    policy = row.get("policy_id", "")
    if scenario == "prefill_decode" and model == "llama31_8b_instruct":
        return policy in {"p00", "p01", "p02", "p03", "p04"}
    if scenario == "prefill_decode" and model == "llama2_7b_chat":
        return policy == "b8o64000"  # dense-BF16 anchor retained by canonical 056.
    return False


def render(model: str, scenario: str) -> None:
    directory = OUT / model / scenario
    rows = read(directory / "data/complete_results.csv")
    fields = list(rows[0])
    uniform = [row for row in rows if is_uniform(model, scenario, row)]
    ours = [row for row in rows if not is_uniform(model, scenario, row)]
    label_field = "policy" if "policy" in fields else "policy_id"
    uniform.sort(key=lambda row: key(row[label_field]))
    ours.sort(key=lambda row: key(row[label_field]))

    def table(items: list[dict[str, str]]) -> str:
        output = ["| " + " | ".join(fields) + " |",
                  "| " + " | ".join("---" for _ in fields) + " |"]
        for row in items:
            output.append("| " + " | ".join(row.get(field) or "—" for field in fields) + " |")
        return "\n".join(output)

    source_path = "`data/complete_results.csv`"
    text = f"""# Compact retained-result summary: {model} / {scenario}

Each policy occupies exactly one row. The source-long per-dataset records remain in `data/` where applicable; this file is their presentation-oriented pivot.

## Uniform references

{table(uniform) if uniform else "No separately retained uniform row in this canonical track; the dense-BF16 anchor is listed above when available."}

## Ours / solved policy points

{table(ours)}

The authoritative machine-readable table is {source_path}. Empty cells were not measured by the retained source experiment and are not inferred here.
"""
    (directory / "results/source_summary.md").write_text(text)


def main() -> None:
    for model in ("llama2_7b_chat", "llama31_8b_instruct"):
        for scenario in ("prefill_only", "prefill_decode"):
            render(model, scenario)


if __name__ == "__main__":
    main()
