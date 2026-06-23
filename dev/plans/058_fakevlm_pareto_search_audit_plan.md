# FakeVLM Pareto Search Audit Plan

## Summary
- Create `artifacts/debug/025_fakevlm_pareto_search_audit` as an independent small-scale search audit for FakeVLM.
- Generate random, neighborhood, and suspicious-module mixed policies, then validate each with real full-model prefill speed and fixed-random 20% FakeClue accuracy.
- Compare searched policies against the existing `024_fakevlm_prefill_global_pareto` validated frontier.

## Implementation
- Build a fixed 1000/5000 FakeClue subset manifest using seed `20260619`; all policies use this same subset.
- Generate about 60 policies for batch 16:
  - 30 random policies across conservative to aggressive replacement ratios.
  - 20 neighborhood policies mutated from representative 024 Pareto points.
  - 10 suspicious-module policies mutating high-risk modules around selected 024 parents.
- Implement a combined validator that loads one policy once, applies real runtime backends, measures real prefill E2E speed, then evaluates 20% subset accuracy.
- Add a launcher that assigns policy jobs across 6 GPUs and supports resume/skip.
- Summarize searched non-dominated points, gap to 024 frontier, and speed-vs-accuracy plots.

## Verification
- Run syntax checks for new scripts.
- Generate subset and policies.
- Run a tiny GPU smoke test before starting the full 6-GPU audit.
- Start full validation and monitor initial logs for successful progress.

## Assumptions
- Candidate methods match 024 prefill-only: `dense_bf16`, `dense_nvfp4`, `sparse_bf16`, `sparse_nvfp4`.
- The primary comparison batch is `batch_size=16`.
- Real speed is measured with warmup 3 and 10 iterations; quality uses 20% subset accuracy with batch size 8.
