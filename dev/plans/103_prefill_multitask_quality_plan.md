# Prefill-only multi-task quality evaluation

## Goal

Create a debug-only, reproducible Llama2-7B-Chat and Llama3.1-8B-Instruct prefill-only quality suite. It evaluates every policy currently published in each model's prefill-only result table on WikiText, Winogrande, ARC-Easy and MMLU, excluding C4.

## Plan

1. Build a policy manifest from the final prefill-only rows, mapping uniform methods to prepared artifacts and ours rows to their frozen policy JSON. → verify 14 Llama2 and 12 Llama3.1 entries.
2. Implement a Transformers/lm-eval HFLM evaluator that injects prepared compression weights according to a uniform or per-layer prefill policy. → verify with a small dense/uniform/ours smoke suite on both models.
3. Schedule all policy/task pairs on explicitly supplied GPUs with resumable per-task outputs and raw lm-eval provenance. → verify every policy has four full result JSON files.
4. Produce debug-only long/wide CSV and Markdown summaries joined with existing speed/recommendation metadata. → do not modify either published `pareto_summary` directory.
