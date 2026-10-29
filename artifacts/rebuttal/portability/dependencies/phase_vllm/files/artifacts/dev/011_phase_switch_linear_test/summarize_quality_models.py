#!/usr/bin/env python3
"""Summarize lightweight quality checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = [
    "bf16",
    "dense_nvfp4",
    "w4a16_ours",
    "phase_switch_overlap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def generation_match(model: dict, baseline: dict) -> dict:
    same = 0
    total = 0
    per_prompt = []
    for cur, base in zip(model["generations"], baseline["generations"]):
        cur_ids = cur["token_ids"]
        base_ids = base["token_ids"]
        n = min(len(cur_ids), len(base_ids))
        matches = sum(1 for i in range(n) if cur_ids[i] == base_ids[i])
        same += matches
        total += n
        per_prompt.append({
            "prompt": cur["prompt"],
            "matched": matches,
            "compared": n,
            "match_rate": matches / n if n else None,
        })
    return {
        "matched": same,
        "compared": total,
        "match_rate": same / total if total else None,
        "per_prompt": per_prompt,
    }


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    rows = {label: load(result_dir / f"{label}_quality.json")
            for label in LABELS}
    baseline = rows["bf16"]
    summary_rows = []
    for label in LABELS:
        data = rows[label]
        rel = {
            "label": label,
            "avg_nll": data["avg_nll"],
            "perplexity": data["perplexity"],
            "ppl_ratio_vs_bf16": data["perplexity"] / baseline["perplexity"],
            "avg_nll_delta_vs_bf16": data["avg_nll"] - baseline["avg_nll"],
            "generation_match_vs_bf16": generation_match(data, baseline),
        }
        summary_rows.append(rel)

    summary = {
        "metric": "prompt-logprob perplexity on fixed continuations; greedy token match against bf16 baseline",
        "num_examples": baseline["num_examples"],
        "num_scored_tokens": baseline["num_scored_tokens"],
        "rows": summary_rows,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) +
                        "\n")

    lines = [
        "# Phase Switch Quality Sanity Check",
        "",
        "Prompt-logprob perplexity uses fixed short continuations. Lower is better.",
        "Greedy match compares 64 generated tokens per prompt against the BF16 baseline.",
        "",
        "| Model | Avg NLL | PPL | PPL / BF16 | NLL delta | Greedy token match |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        match = row["generation_match_vs_bf16"]["match_rate"]
        lines.append(
            f"| {row['label']} | {row['avg_nll']:.4f} | "
            f"{row['perplexity']:.4f} | {row['ppl_ratio_vs_bf16']:.4f} | "
            f"{row['avg_nll_delta_vs_bf16']:.4f} | "
            f"{match:.4f} |")
    lines.extend(["", "## Generation Samples", ""])
    for prompt_idx, base_gen in enumerate(baseline["generations"], start=1):
        lines.extend([f"### Sample {prompt_idx}", "",
                      f"Prompt: `{base_gen['prompt']}`", ""])
        for label in LABELS:
            text = rows[label]["generations"][prompt_idx - 1]["output"]
            text = text.replace("\n", "\\n")
            if len(text) > 180:
                text = text[:177] + "..."
            lines.append(f"- {label}: {text!r}")
        lines.append("")
    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
