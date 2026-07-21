#!/usr/bin/env python3
"""Summarize new real-vLLM prefill-decoding NLL results without touching old artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEBUG = ROOT / "artifacts/debug/044_llama_prefill_decode_vllm_nll"
RESULTS = DEBUG / "results"


def classify(name: str) -> str:
    if name in {"dense_bf16", "dense_nvfp4", "marlin_nvfp4", "sparse_bf16", "sparse_nvfp4"}:
        return "uniform"
    if name == "max_speed":
        return "ours-max-speed"
    return "ours-formal-pareto"


def main() -> None:
    rows: list[dict[str, object]] = []
    for scenario in ("llama2_prefill_decode", "llama31_prefill_decode"):
        for path in sorted((RESULTS / scenario).glob("*.json")):
            if path.name.endswith(".phase_trace.json"):
                continue
            data = json.loads(path.read_text())
            runtime = data["runtime"]
            phase = bool(runtime["phase_hetero"])
            rows.append({
                "model": scenario.removesuffix("_prefill_decode"),
                "policy": path.stem,
                "family": classify(path.stem.split("_runtime")[0]),
                "avg_nll": float(data["avg_nll"]),
                "perplexity": float(data["perplexity"]),
                "token_count": int(data["token_count"]),
                "phase_hetero": phase,
                "phase_trace": runtime.get("phase_trace", ""),
                "trace_events": json.dumps(runtime.get("phase_trace_events", {}), sort_keys=True),
                "result_path": str(path.relative_to(ROOT)),
            })
    rows.sort(key=lambda row: (str(row["model"]), str(row["family"]), str(row["policy"])))
    fields = list(rows[0])
    (DEBUG / "report").mkdir(exist_ok=True)
    with (DEBUG / "report" / "real_vllm_prefill_decode_nll.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    lines = ["# Real vLLM teacher-forced prefill-decoding NLL", "",
             "All rows use 2048 prefill tokens + 80 teacher-forced decode tokens on 32 WikiText blocks (2560 scored tokens).",
             "This report is isolated under debug 044 and does not replace historical proxy-NLL artifacts.", "",
             "Formal-point aliases: point_000 is policy-identical to dense BF16 for both models; Llama2 point_011 is policy-identical to its max-speed result.", "",
             "| model | family | policy | avg NLL | perplexity | phase trace |", "|---|---|---|---:|---:|---|"]
    for row in rows:
        trace = "yes" if row["phase_hetero"] else "—"
        lines.append(f"| {row['model']} | {row['family']} | {row['policy']} | {row['avg_nll']:.6f} | {row['perplexity']:.4f} | {trace} |")
    (DEBUG / "report" / "real_vllm_prefill_decode_nll.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
