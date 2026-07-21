# Llama3.1 prefill speed decomposition debug plan

## Goal

Separate Llama3.1-8B-Instruct prefill-only high-sparsity speed error into:

1. per-linear KernelLatencyPredictor error; and
2. residual raw-kernel-sum to real phase-vLLM E2E error.

This is a debug experiment. It must not overwrite the 058 canonical results.

## Scope and assumptions

- Scenario is unchanged: prefill-only, batch 8, input length 2048.
- Reuse 058 policies and the same phase-vLLM exporter/benchmark.
- Measure exactly the four repeated Llama3 fused linear shapes and five supported methods; no `--prune`, no new sparse weights.
- Use the exact microbenchmark sum for fixed diagnostic policies only. Do not alter the main solver before evidence identifies the faulty layer.

## Steps

1. Create `artifacts/debug/059_llama31_prefill_speed_decomposition/` and inventory the 058 action-support predictions by shape/method.
   - Verify: identify measured versus predicted local entries and the set requiring exact microbenchmark.
2. Benchmark missing Llama3 module shapes through the existing CUTLASS modeling/audit path and assemble an exact per-shape local-latency table.
   - Verify: each policy method/action has an exact local latency or an explicit unsupported status.
3. Compare predicted local sums and exact local sums for uniform anchors plus high-sparse policies (`p014`, bridge/high-sparse probes).
   - Verify: report absolute/relative local prediction error by method composition.
4. Compare exact local sums against independent phase-vLLM E2E measurements for the same policies.
   - Verify: report residual E2E composition overhead and determine whether it correlates with sparse-BF16/NVFP4 mixture.
5. Write a concise diagnosis and a bounded follow-up recommendation: local predictor repair, E2E calibrator repair, or both.
   - Verify: preserve 058 results and link all source artifacts.
