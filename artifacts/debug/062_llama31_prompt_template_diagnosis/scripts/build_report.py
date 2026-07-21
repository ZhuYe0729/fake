#!/usr/bin/env python3
"""Compare legacy and native-template diagnosis outputs."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu


ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "artifacts/debug/062_llama31_prompt_template_diagnosis"
TASKS = ("cnn_dm_1000", "dsum", "IWSLT")


def read(template: str, task: str) -> list[dict]:
    return [json.loads(line) for line in (EXP / "outputs" / template / task / "answers.jsonl").read_text().splitlines()]


def continuation_markers(text: str) -> list[str]:
    # The terminal <|eot_id|> is an expected stop token, not a role continuation.
    markers = ("Human:", "Assistant:", "<|start_header_id|>", "<|end_header_id|>")
    return [marker for marker in markers if marker in text]


def score(task: str, rows: list[dict]) -> dict[str, float | int]:
    predictions = [row["text"] if row["text"].strip() else "." for row in rows]
    references = [row["reference"] for row in rows]
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    result: dict[str, float | int] = {"samples": len(rows), "avg_new_tokens": sum(row["new_tokens"] for row in rows) / len(rows),
                                      "avg_input_tokens": sum(row["input_tokens"] for row in rows) / len(rows),
                                      "role_marker_continuations": sum(bool(continuation_markers(row["text"])) for row in rows),
                                      "finish_reasons": dict(Counter(str(row["finish_reason"]) for row in rows))}
    if task == "IWSLT": result["sacre_bleu"] = corpus_bleu(predictions, [references]).score
    else: result["rougeL_percent"] = 100 * sum(rouge.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)) / len(rows)
    return result


def main() -> None:
    report = EXP / "report"; report.mkdir(parents=True, exist_ok=True)
    rows = {template: {task: read(template, task) for task in TASKS} for template in ("legacy", "native")}
    protocol_check = {}
    for task in TASKS:
        legacy_ids = [(row["question_id"], row["reference"]) for row in rows["legacy"][task]]
        native_ids = [(row["question_id"], row["reference"]) for row in rows["native"][task]]
        protocol_check[task] = {"same_question_ids_and_references": legacy_ids == native_ids,
                                "samples": len(legacy_ids)}
        if legacy_ids != native_ids:
            raise RuntimeError(f"{task}: template arms do not share the fixed PMPD subset")
    metrics = {template: {task: score(task, rows[template][task]) for task in TASKS} for template in ("legacy", "native")}
    manifests = {template: json.loads((EXP / "outputs" / template / "run_manifest.json").read_text())
                 for template in ("legacy", "native")}
    (report / "metrics.json").write_text(json.dumps({"metrics": metrics, "protocol_check": protocol_check}, indent=2) + "\n")
    lines = ["# Llama3.1-8B-Instruct dense-BF16 prompt-template diagnosis", "",
             "This is a diagnostic only. It does not replace legacy/common main-experiment results.",
             "Both arms use identical fixed PMPD prefixes (100 examples/task), vLLM BF16, greedy decoding, max_new_tokens=256, max_input_tokens=3840, and explicit EOS+EOT stops. Only prompt construction differs.",
             f"The tokenizer has eos_token_id={manifests['legacy']['eos_token_id']} and eot_token_id={manifests['legacy']['eot_token_id']}; both arms therefore use the same stop_token_ids={manifests['legacy']['stop_token_ids']}.", "",
             "| task | legacy metric | native metric | legacy avg new tokens | native avg new tokens | legacy marker continuations | native marker continuations |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        key = "sacre_bleu" if task == "IWSLT" else "rougeL_percent"
        left, right = metrics["legacy"][task], metrics["native"][task]
        lines.append(f"| {task} | {left[key]:.3f} | {right[key]:.3f} | {left['avg_new_tokens']:.2f} | {right['avg_new_tokens']:.2f} | {left['role_marker_continuations']} | {right['role_marker_continuations']} |")
    lines += ["", "`role_marker_continuations` excludes a terminal `<|eot_id|>` because it is the expected shared stop token.", "",
              "## Finish reasons", ""]
    for task in TASKS:
        lines.append(f"- {task}: legacy {metrics['legacy'][task]['finish_reasons']}; native {metrics['native'][task]['finish_reasons']}")
    lines += ["", "## Conclusion", "",
              "On the identical fixed subset, the native Llama3 chat template improves all three primary metrics and eliminates the widespread length-capped behavior of the legacy prompt.",
              "This supports prompt-template mismatch as a major explanation for Llama3.1's low *cross-model absolute* generation scores under the legacy/common PMPD protocol. It does not invalidate within-Llama3 compression comparisons, because those main results hold that legacy protocol fixed across BF16, uniform, and ours.",
              "The native-template numbers are diagnostic only and must not replace the legacy/common main table.", "",
              "## Fixed-subset check", ""]
    for task, check in protocol_check.items():
        lines.append(f"- {task}: same question IDs and references across arms = {check['same_question_ids_and_references']} ({check['samples']} samples).")
    lines += ["", "## Paired examples", ""]
    for task in TASKS:
        legacy = {row["question_id"]: row for row in read("legacy", task)}
        native = {row["question_id"]: row for row in read("native", task)}
        lines += [f"### {task}", ""]
        for question_id in sorted(legacy)[:3]:
            lines += [f"- id {question_id}; legacy: `{legacy[question_id]['text'][:360]}`", f"  native: `{native[question_id]['text'][:360]}`"]
    (report / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
