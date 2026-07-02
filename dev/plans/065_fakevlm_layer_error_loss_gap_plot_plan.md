# FakeVLM layer error vs loss gap plot plan

## Goal

Draw a focused FakeVLM debug figure showing that, under the same compression method, local per-layer output errors can be close while final quality/loss impact differs substantially.

## Assumptions

- Reuse existing FakeVLM artifacts from `artifacts/debug/024_fakevlm_prefill_global_pareto` first.
- Do not rerun GPU evaluation unless required; existing local error, fitted quality-cost, and measured loss files are sufficient for this visualization.
- Treat `assistant_answer_token_nll_v2_active_prefix_aligned` as the valid loss definition.

## Steps

1. Inspect available FakeVLM local-error and loss artifacts.
   - Verify: required CSV files exist and contain 224 language linear modules.
2. Create a new numbered debug directory with an analysis/plot script.
   - Verify: script can regenerate CSV summaries and plots from existing artifacts.
3. Produce a figure and short README explaining the evidence and limitations.
   - Verify: generated outputs are present under the new debug directory.
