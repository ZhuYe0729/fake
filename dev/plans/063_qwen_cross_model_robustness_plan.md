# Qwen Cross-Model Robustness Benchmark Plan

## Summary
- Create `artifacts/debug/027_qwen_cross_model_robustness/` for Qwen3.5 cross-model speed robustness tests.
- Measure multiple Qwen3.5 sizes with the same workload and method grid used by the FakeVLM cross-workload experiment.
- Produce tables that show uniform methods can be model-size dependent, while `our_linear_hybrid` keeps a stronger average.

## Key Changes
- Add debug-only scripts under `artifacts/debug/027_qwen_cross_model_robustness/scripts/`.
- Run synthetic-token Qwen E2E latency with prefill and KV-cache decode breakdown:
  - `prefill_ms`
  - `decode_avg_ms`
  - `decode_total_ms`
  - `e2e_ms`
- Default models:
  - `Qwen3.5-0.8B`
  - `Qwen3.5-2B`
  - `Qwen3.5-4B`
  - `Qwen3.5-9B`
- Default workloads:
  - `prefill_only`: `batch_size=16,input_tokens=1024,output_tokens=0`
  - `normal_01`: `batch_size=1,input_tokens=16384,output_tokens=32`
  - `normal_02`: `batch_size=1,input_tokens=16384,output_tokens=256`
- Compare fixed methods:
  - `dense_bf16`
  - `uniform_dense_nvfp4`
  - `uniform_sparse_bf16`
  - `uniform_sparse_nvfp4`
  - `uniform_marlin_weight_only`
  - `uniform_dense_nvfp4_prefill_marlin_decode`
  - `our_linear_hybrid`
- Build `our_linear_hybrid` per model and workload using the offline predictor policy over:
  - `dense_bf16`
  - `dense_nvfp4`
  - `sparse_bf16`
  - `sparse_nvfp4`
  - `marlin_nvfp4`

## Outputs
- `policies/{model}/{scenario}/our_linear_hybrid/policy.json|csv`
- `speed/qwen_cross_model_raw.csv`
- `summary/model_workload_method_table.csv|md`
- `summary/model_average_table.csv|md`
- `summary/cross_model_transfer.csv|md`
- `summary/analysis.md`

## Test Plan
- Static:
  - `python -m py_compile artifacts/debug/027_qwen_cross_model_robustness/scripts/*.py`
  - `bash -n artifacts/debug/027_qwen_cross_model_robustness/scripts/*.sh`
- Smoke:
  - run `Qwen3.5-0.8B` with `prefill_only` for `dense_bf16` and `our_linear_hybrid`;
  - run one decode workload with `OVERRIDE_OUTPUT_TOKENS=2`.
- Full:
  - run all methods across all default models and workloads with task-level GPU parallelism;
  - keep one benchmark task per GPU.

## Assumptions
- This experiment is speed-only; no accuracy modeling or dataset calibration is needed.
- `Qwen3.5-27B` is excluded from the default grid because it likely needs model parallelism and would change the single-GPU task-level timing protocol.
- Conversion latency is included only through the offline policy when WA and weight-only NVFP4 layouts require real conversion work.
