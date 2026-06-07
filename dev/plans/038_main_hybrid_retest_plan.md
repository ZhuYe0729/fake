# Main Hybrid Policy Retest Plan

## Summary
- Re-test Llama-2-7B, Llama-3.1-8B, and Qwen3.5-9B for `prefill_only` and `normal_01`.
- Save results under `artifacts/results/main/001_hybrid_policy_retest/`.
- Produce single-backend, manual-policy, and predictor-policy results with both linear-module aggregate latency and full-model E2E latency.

## Implementation
- Add one orchestrator script that enumerates compressible linear groups, benchmarks six manual candidates, generates predictor policies, and runs full-model E2E.
- Split output by method family and scenario:
  - `manual/{scenario}/`
  - `pred/{scenario}/`
  - `single/{method}/{scenario}/`
  - `comparison/`
- Keep leaf-level README/CSV/JSON outputs so interrupted runs can be inspected and resumed.

## Verification
- Compile new script with `python -m py_compile`.
- Run GPU benchmarks in the local `cospaq` environment.
- Check `comparison/full_e2e_summary.csv`, `comparison/linear_latency_summary.csv`, and `comparison/manual_vs_pred_policy_diff.csv`.

## Assumptions
- `normal_01` is `batch_size=1,input_tokens=16384,output_tokens=32`.
- `prefill_only` is `batch_size=16,input_tokens=1024,output_tokens=0`.
- `marlin_nvfp4` is the W4A16 weight-only method.
- Full-model E2E is the final metric; linear aggregate latency is used for strategy explanation.
