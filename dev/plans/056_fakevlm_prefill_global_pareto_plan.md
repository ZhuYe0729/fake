# 056 FakeVLM Prefill Global Pareto Plan

## Summary
- Create `artifacts/debug/024_fakevlm_prefill_global_pareto/`.
- Follow the `018_llama2_prefill_global_pareto` workflow for FakeVLM:
  1. build FakeVLM accuracy/quality modeling data,
  2. build FakeVLM speed modeling data,
  3. generate quality-constrained Pareto policies,
  4. validate selected policies with real FakeVLM speed and accuracy runs,
  5. summarize tables, plots, and analysis.
- Keep the compression target to the existing FakeVLM language-model linear layer set.

## Key Changes
- Use the existing FakeVLM compressible-module selection: 224 language-model `nn.Linear` modules.
- Candidate methods for optimized Pareto policies:
  - `dense_bf16`
  - `dense_nvfp4`
  - `sparse_bf16`
  - `sparse_nvfp4`
- Do not include `marlin_weight_only` or `dense_nvfp4_prefill_marlin_decode` as optimized Pareto candidates unless separate per-module quality modeling data is added for them.
- Accuracy/quality modeling:
  - collect per-module local output errors for each candidate method;
  - generate stratified mixed policies;
  - measure policy-level FakeVLM quality on FakeClue;
  - fit a multiplicative global/layer/type coefficient model similar to `018`.
- Speed modeling:
  - reuse the existing FakeVLM `021` per-shape/per-method prefill latency data;
  - build one Pareto frontier per batch size: `1, 2, 4, 8, 16`.
- Validation:
  - apply selected Pareto policies to real FakeVLM runtime kernels;
  - measure real prefill latency;
  - measure FakeClue `global_accuracy`;
  - join modeled cost, measured speed, and measured accuracy.

## Test Plan
- Static checks:
  - `python -m py_compile artifacts/debug/024_fakevlm_prefill_global_pareto/scripts/*.py`
  - `bash -n` for shell launchers if added.
- Smoke:
  - collect local errors for a small module/sample subset;
  - generate a few stratified policies;
  - run quality fitting on smoke rows;
  - build one batch Pareto frontier;
  - validate one or two policies on a small FakeClue sample.
- Full:
  - collect complete local errors for all 224 modules;
  - measure enough stratified policy quality rows for coefficient fitting;
  - build Pareto frontiers for batch `1,2,4,8,16`;
  - validate selected policies and summarize final artifacts.

## Assumptions
- This is prefill-only, matching the current FakeVLM speed study.
- Vision tower, multimodal projector, embeddings, norms, attention softmax, and output head are not compressed.
- Policy generation uses modeled quality cost, not raw local error alone.
