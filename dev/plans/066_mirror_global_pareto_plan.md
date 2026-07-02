# 066 MIRROR Layer-Heterogeneous Pareto Plan

## Summary
- Create `artifacts/debug/030_mirror_global_pareto/`.
- Implement a MIRROR single-forward workflow similar to FakeVLM `024`: local quality proxy, policy quality modeling, speed modeling, constrained Pareto optimization, selected-policy validation, and report plots.
- Treat the original uncompressed MIRROR runtime as a first-class candidate and baseline.

## Key Changes
- Candidate methods:
  - `dense_default`: original MIRROR dense runtime, no layer replacement.
  - `dense_bf16`: BF16 dense reference, no compression.
  - `dense_nvfp4`
  - `sparse_bf16`
  - `sparse_nvfp4`
- Compress only `select_compressible_modules(model, "mirror")`, currently DINOv3 backbone transformer Linear layers.
- Use MIRROR classification CE/NLL delta for quality modeling and report measured `bal_acc`, `auc`, and `ap`.
- Prefer more diverse policy samples on a fixed stratified subset for quality-model fitting; reserve full-dataset downstream metrics for selected Pareto validation.
- Use captured MIRROR forward inputs for local errors and standalone per-layer latency costs.
- Validate selected policies with real full-detector forward speed and real dataset metrics.

## Test Plan
- Static:
  - `python -m py_compile artifacts/debug/030_mirror_global_pareto/scripts/*.py`
  - `bash -n artifacts/debug/030_mirror_global_pareto/scripts/*.sh`
- Smoke after model/data transfer:
  - use small sample/module/policy limits to run local errors, speed modeling, policy quality, coefficient fitting, cost table, optimization, selected validation, and report plot.
- Full:
  - all MIRROR selected Linear layers have complete candidate rows.
  - quality-model training can use a representative fixed subset with more policy samples.
  - selected Pareto policies have full forward latency and real classification metrics.
  - report outputs include CSV and PNG/PDF Pareto figures.

## Assumptions
- MIRROR is a single classification-forward workload, not a prefill/decode workload.
- `dense_default` means the current default MIRROR dense loading path with no forced BF16 conversion.
- Model/data default paths are local-machine paths under `/home/agent/wja/data`.
