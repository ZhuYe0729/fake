# Llama2 Prefill Global-Coeff Pareto Plan

## Summary
- Create `artifacts/debug/018_llama2_prefill_global_pareto` without overwriting earlier debug runs.
- Rebuild the Llama2 prefill-only Pareto workflow with the 017 multiplicative global-coefficient quality proxy.
- Validate every unique frontier point on local GPUs, using at most one GPU per process and allocating GPUs in `7,6,5,4,3,2` order.
- Use real compressed artifacts and real runtime kernels for quality and latency validation; do not alias Marlin W4A16 quality to dense NVFP4.

## Implementation
- Copy the 008 Pareto workflow into the new 018 debug directory and retarget it to 018-local outputs.
- Build module-method costs from real local-error rows plus 017 `final_layer_type` coefficients:
  - `dense_bf16`: zero quality cost.
  - `dense_nvfp4`, `sparse_bf16`, `sparse_nvfp4`: `global_coef * layer_coef[layer] * type_coef[type] * local_error`.
  - `marlin_nvfp4`: only include as a Pareto candidate when a real Marlin-specific local-error and coefficient source is provided; otherwise keep it as a real uniform baseline only.
- Update validation scripts so mixed policies load each selected module from the method-specific `prepared/<method>/model.pt` artifact and install the matching runtime kernel.
- Add local GPU launcher scripts for full E2E and quality validation over all unique Pareto points.

## Visualization
- Main plot: validated median E2E speedup vs validated NLL delta.
- Show uniform baselines with real measured quality/latency, including Marlin W4A16 when available.
- Mark noisy latency points using fixed QC rules; do not hide points manually to make the conclusion stronger.
- Add a dominance table that states which baselines are dominated by key Pareto points.

## Verification
- Run syntax checks for all 018 scripts.
- Run CPU-side cost-table, optimization, summary, and plot generation.
- Smoke-test one policy containing Marlin before full validation if Marlin is enabled as a candidate.
- Full validation writes joined E2E/quality results and a final analysis markdown.

## Assumptions
- This plan targets prefill-only only; normal decode scenarios are out of scope.
- Existing 017 coefficients are trusted for methods they explicitly cover.
- Marlin is never treated as dense NVFP4 for quality modeling; if no Marlin proxy is available, it is excluded from the optimized candidate set and reported as a uniform baseline.
