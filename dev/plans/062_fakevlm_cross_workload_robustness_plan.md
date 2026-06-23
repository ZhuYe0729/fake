# FakeVLM Cross-Workload Robustness Plan

## Summary
- Create `artifacts/debug/026_fakevlm_cross_workload_robustness/`.
- Extend FakeVLM speed tests from prefill-only to three workloads:
  - `prefill_only`: `batch_size=16,input_tokens=1024,output_tokens=0`
  - `normal_01`: `batch_size=1,input_tokens=16384,output_tokens=32`
  - `normal_02`: `batch_size=1,input_tokens=16384,output_tokens=256`
- Produce fixed-method summary tables showing that uniform methods are not cross-workload robust, while the per-linear workload-aware hybrid has better average behavior.

## Key Changes
- Add debug-only scripts under `artifacts/debug/026_fakevlm_cross_workload_robustness/scripts/`.
- Measure real FakeVLM E2E speed with prefill/decode breakdown:
  - `prefill_ms`
  - `decode_avg_ms`
  - `decode_total_ms`
  - `e2e_ms`
- Compare fixed method columns:
  - `dense_bf16`
  - `uniform_dense_nvfp4`
  - `uniform_sparse_bf16`
  - `uniform_sparse_nvfp4`
  - `uniform_marlin_weight_only`
  - `uniform_dense_nvfp4_prefill_marlin_decode`
  - `our_linear_hybrid`
- Build `our_linear_hybrid` per workload with per-linear backend choices over:
  - `dense_bf16`
  - `dense_nvfp4`
  - `sparse_bf16`
  - `sparse_nvfp4`
  - `marlin_nvfp4`
- Select the minimum legal per-linear cost:
  - `prefill_latency + output_tokens * decode_latency + applicable_conversion_latency`
  - conversion latency is included only for real layout/weight-format transitions, especially WA `dense_nvfp4` and weight-only `marlin_nvfp4`.
- Use only task-level GPU parallelism on GPUs `0,1,2,3`; never run two speed tasks on the same GPU at the same time.
- Use conda environment `cospaq`.

## Outputs
- `policies/{scenario}/{method}/policy.json|csv`
- `speed/e2e_speed_raw.csv`
- `summary/workload_method_table.csv`
- `summary/workload_method_table.md`
- `summary/cross_workload_transfer.csv`
- `summary/cross_workload_transfer.md`
- `summary/analysis.md`

## Test Plan
- Static:
  - `python -m py_compile artifacts/debug/026_fakevlm_cross_workload_robustness/scripts/*.py`
  - `bash -n artifacts/debug/026_fakevlm_cross_workload_robustness/scripts/*.sh`
- Smoke:
  - run `dense_bf16`, `uniform_marlin_weight_only`, and `our_linear_hybrid`;
  - use `sample_limit=2,warmup=1,iters=2`;
  - optionally override output tokens to `2` for decode-loop validation.
- Full:
  - run all seven methods across all three workloads with `warmup=3,iters=10`;
  - verify every workload has a dense BF16 baseline and every method has `e2e_ms` plus `speedup_vs_dense_bf16`.

## Assumptions
- Current machine has directly accessible GPUs; no SLURM setup is needed.
- If GPU access is blocked by sandboxing, run required checks with escalated permissions.
- This experiment optimizes speed only; quality is referenced from existing FakeVLM uniform/Pareto artifacts rather than remeasured here.
- Compression remains restricted to FakeVLM language-model `nn.Linear` modules.
