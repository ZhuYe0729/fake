# 059 FakeVLM Prediction Validation Plan

## Summary
- Add prediction-versus-measurement validation for the 40 selected FakeVLM Pareto policies.
- Compare predicted NLL, measured NLL, FakeClue accuracy, per-linear latency, and end-to-end latency.
- Write independent artifacts under `artifacts/debug/024_fakevlm_prefill_global_pareto/prediction_vs_actual/` without replacing existing reports.

## Key Changes
- Measure assistant-answer-token teacher-forcing NLL for all selected policies, using one dense baseline followed by eight parallel GPU shards.
- Join predicted quality cost, measured NLL, and measured FakeClue accuracy; report raw and clipped NLL deltas separately.
- Compare `021` latency-model outputs against `021/manual_profile` for all 60 batch/shape/backend combinations, separating exact measured lookups from model predictions.
- Estimate per-batch non-linear latency as dense measured E2E minus dense measured linear sum, then predict each policy E2E as this constant plus its predicted linear sum.
- Generate CSV, Markdown, PNG, and PDF comparison artifacts with `_prediction_vs_actual` filenames.

## Test Plan
- Run Python `py_compile` and shell `bash -n` checks.
- Require 40/40 selected speed, accuracy, and NLL rows, plus 60/60 single-linear comparison rows.
- Verify every policy contains 224 linears and every shape/backend has both measured and latency-model values.
- Verify existing `report/` artifacts remain unchanged.

## Assumptions
- Quality loss uses FakeClue assistant-answer-token NLL, matching the fitted quality model.
- The non-linear E2E component is a per-batch dense residual constant.
- Existing `021/manual_profile` rows are the real single-linear measurements; no new speed benchmark is run.
