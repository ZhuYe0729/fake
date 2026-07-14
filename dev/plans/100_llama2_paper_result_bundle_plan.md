# Llama2-7B-Chat paper result bundle

## Goal

Create the same paper-facing result bundle used for Llama-3.1 under
`artifacts/exports/vllm/ours/llama2-7b-chat/pareto_summary/`. It must retain
every measured uniform method and every ours point that has the corresponding
task score, including each scenario's max-speed endpoint; it must not silently
substitute predicted data for measured data.

## Plan

1. Read the final measured source tables for prefill-only ARC and prefill-decode
   three-task results, plus their measured NLL/speed records. → verify row
   counts and source fields before aggregation.
2. Add a small deterministic aggregation/plot script that writes one CSV, one
   Markdown table, ARC and three PMPD task Pareto figures, and a PMPD NLL figure.
   → verify the emitted rows reproduce source speed and score values.
3. Mark recommended candidates only as suggestions: near-lossless, balanced,
   fast, and max-speed; retain all other measured points without filtering.
   → verify every measured ours point is present and labels distinguish any
   screened/stalled intermediate measurements.
4. Run the script, inspect its summary and figures, and record this follow-up
   implementation in `dev/impls/100_llama2_paper_result_bundle_impl.md`.

## Scope notes

- Prefill-only source: `037/.../arc_challenge/report/arc_challenge_speed_summary.csv`.
- Prefill-decode core source: `035/.../task_quality_all/summary.csv`; intermediate
  points 34/36/37/38 are included but visibly marked as screened-stall measurements.
- No new GPU experiments are needed.
