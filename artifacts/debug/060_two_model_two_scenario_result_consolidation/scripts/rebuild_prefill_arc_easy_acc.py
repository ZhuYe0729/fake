#!/usr/bin/env python3
"""Synchronize current prefill-only consolidations after ARC-Easy switches to acc."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/debug/060_two_model_two_scenario_result_consolidation"
L2_SOURCE = ROOT / "artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/llama2_7b_chat/pareto/paper"
L3_SOURCE = ROOT / "artifacts/debug/061_llama31_prefill_warmed_speed_revalidation/llama31_8b_instruct/report"


def markdown_table(rows: list[dict[str, str]], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |",
             "| " + " | ".join("---" for _ in fields) + " |"]
    lines += ["| " + " | ".join(row.get(field) or "—" for field in fields) + " |" for row in rows]
    return "\n".join(lines)


def copy_figures(source: Path, destination: Path, pattern: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.glob(pattern):
        shutil.copy2(path, destination / path.name)


def rebuild_llama2() -> None:
    destination = OUT / "llama2_7b_chat/prefill_only"
    data = destination / "data"
    for source_name, destination_name in (("all_methods_measured.csv", "complete_results.csv"),
                                          ("uniform_references.csv", "uniform_references.csv"),
                                          ("ours_measured_pareto.csv", "ours_measured_pareto.csv")):
        shutil.copy2(L2_SOURCE / source_name, data / destination_name)
    copy_figures(L2_SOURCE, destination / "figures", "pareto_speed_vs_*.png")
    rows = list(csv.DictReader((data / "complete_results.csv").open()))
    fields = list(rows[0])
    summary = "\n".join([
        "# llama2_7b_chat: prefill-only", "",
        "B=8, input=2048. Speed and quality are retained real phase-vLLM measurements.",
        "ARC-Easy uses `acc`; ARC-Challenge uses `acc_norm`.", "",
        "## Complete measured-result table", "", markdown_table(rows, fields), "",
        "## Figures", "",
        *[f"- [figures/{path.name}](figures/{path.name})" for path in sorted((destination / "figures").glob("pareto_speed_vs_*.png"))], "",
        "The machine-readable source is `data/complete_results.csv`.", "",
    ])
    (destination / "summary.md").write_text(summary)


def rebuild_llama3() -> None:
    destination = OUT / "llama31_8b_instruct/prefill_only_02"
    data = destination / "data"
    shutil.copy2(L3_SOURCE / "measured_task_pareto.csv", data / "complete_results.csv")
    copy_figures(L3_SOURCE / "task_pareto", destination / "figures", "speed_vs_*.png")
    rows = list(csv.DictReader((data / "complete_results.csv").open()))
    fields = list(rows[0])
    summary = "\n".join([
        "# llama31_8b_instruct: prefill-only (061 warmed revalidation)", "",
        "B=8, input=2048; real phase-vLLM speed, NLL, and downstream measurements.",
        "ARC-Easy uses `acc`; ARC-Challenge uses `acc_norm`.", "",
        "## Complete measured-result table", "", markdown_table(rows, fields), "",
        "The machine-readable source is `data/complete_results.csv`.", "",
    ])
    (destination / "summary.md").write_text(summary)


if __name__ == "__main__":
    rebuild_llama2()
    rebuild_llama3()
