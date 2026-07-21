#!/usr/bin/env python3
"""Rebuild decode complete tables, retaining task-only uniform speed rows."""
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/debug/060_two_model_two_scenario_result_consolidation"
FIELDS = [
    "policy_id", "e2e_median_ms", "ttft_median_ms",
    "measured_speedup_vs_dense", "raw_predicted_speedup_vs_dense",
    "predicted_delta_nll", "cnn_dm_rougeL_percent",
    "cnn_dm_bert_score_percent", "dsum_rougeL_percent",
    "dsum_bert_score_percent", "iwslt_rougeL_percent", "iwslt_sacre_bleu",
]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def table(rows: list[dict[str, str]]) -> str:
    lines = ["| " + " | ".join(FIELDS) + " |",
             "| " + " | ".join("---" for _ in FIELDS) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row.get(field) or "—" for field in FIELDS) + " |")
    return "\n".join(lines)


def rebuild(model: str, caveat: str) -> None:
    scenario = OUT / model / "prefill_decode"
    rows = {row["policy_id"]: dict(row) for row in read(scenario / "data/closure_summary.csv")}
    for item in read(scenario / "data/task_summary_long.csv"):
        row = rows.setdefault(item["policy_id"], {"policy_id": item["policy_id"]})
        # Uniform rows in Llama3 have speed only in the task table.
        row.setdefault("measured_speedup_vs_dense", item.get("measured_speedup_vs_dense", ""))
        row.setdefault("predicted_delta_nll", item.get("predicted_delta_nll", ""))
        if item["dataset"] == "cnn_dm_1000":
            row["cnn_dm_rougeL_percent"] = item["rougeL_percent"]
            row["cnn_dm_bert_score_percent"] = item["bert_score_percent"]
        elif item["dataset"] == "dsum":
            row["dsum_rougeL_percent"] = item["rougeL_percent"]
            row["dsum_bert_score_percent"] = item["bert_score_percent"]
        elif item["dataset"] == "IWSLT":
            row["iwslt_rougeL_percent"] = item["rougeL_percent"]
            row["iwslt_sacre_bleu"] = item["sacre_bleu"]
    ordered = list(rows.values())
    write(scenario / "data/complete_results.csv", ordered)
    source = "artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/llama2_7b_chat" if model.startswith("llama2") else "artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto/llama31_8b_instruct"
    figures = "\n".join(f"- [figures/{p.name}](figures/{p.name})" for p in sorted((scenario / "figures").glob("*.png")))
    summary = f"""# {model}: prefill-decode

B=8, input=2048, output=64; phase-vLLM runtime, BF16 KV cache, chunked prefill disabled.

This is a compact consolidation of `{source}`. No measurement was rerun. `data/complete_results.csv` is the machine-readable version of the full retained task-result table; empty E2E/TTFT fields mean the corresponding source recorded speedup but did not retain the raw latency in this table.

## Complete measured-result table

{table(ordered)}

## Figures

{figures}

## Caveat

{caveat}
"""
    (scenario / "summary.md").write_text(summary)


def main() -> None:
    rebuild("llama2_7b_chat", "The complete table is the measured downstream-task subset of the closure; `data/predicted_points.csv` retains the wider solver candidate set.")
    rebuild("llama31_8b_instruct", "`point_007` is retained as measured data but its source report marks its speed as anomalous; do not use it as an envelope point.")


if __name__ == "__main__":
    main()
