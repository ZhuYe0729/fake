# FakeVLM Prefill Global Pareto

This debug run mirrors the `018_llama2_prefill_global_pareto` workflow for FakeVLM.

## Scope

- Model: FakeVLM.
- Scenario: prefill-only, `model(**inputs, use_cache=False)`.
- Environment: run GPU stages from the project conda env `cospaq` with CUDA 12.8.
- Compression target: FakeVLM language-model linear layers selected by `select_compressible_modules(model, "fakevlm")`.
- Expected selected linear count: 224.
- Pareto candidate methods: `dense_bf16`, `dense_nvfp4`, `sparse_bf16`, `sparse_nvfp4`.
- Batch sizes: `1, 2, 4, 8, 16`.

`marlin_weight_only` and `dense_nvfp4_prefill_marlin_decode` can be reported as uniform baselines from `020`, but are not optimized Pareto candidates in this prefill-only run because this directory does not model their per-module quality cost.

## Workflow

```bash
# 1. Collect local output errors.
python artifacts/debug/024_fakevlm_prefill_global_pareto/scripts/collect_local_errors.py \
  --output-root artifacts/debug/024_fakevlm_prefill_global_pareto

# 2. Generate stratified mixed policies for quality modeling.
python artifacts/debug/024_fakevlm_prefill_global_pareto/scripts/generate_quality_policies.py \
  --output-root artifacts/debug/024_fakevlm_prefill_global_pareto

# 3. Measure FakeVLM quality for those policies.
python artifacts/debug/024_fakevlm_prefill_global_pareto/scripts/validate_policy_quality.py \
  --output-root artifacts/debug/024_fakevlm_prefill_global_pareto \
  --policies stratified

# 4. Fit multiplicative coefficients from measured quality rows.
python artifacts/debug/024_fakevlm_prefill_global_pareto/scripts/fit_quality_model.py \
  --output-root artifacts/debug/024_fakevlm_prefill_global_pareto

# 5. Build per-module quality/latency costs from fitted quality model and 021 latency data.
python artifacts/debug/024_fakevlm_prefill_global_pareto/scripts/build_cost_table.py \
  --output-root artifacts/debug/024_fakevlm_prefill_global_pareto

# 6. Optimize one Pareto frontier per batch size.
python artifacts/debug/024_fakevlm_prefill_global_pareto/scripts/optimize_pareto.py \
  --output-root artifacts/debug/024_fakevlm_prefill_global_pareto

# 7. Select representative points and validate real speed/accuracy.
python artifacts/debug/024_fakevlm_prefill_global_pareto/scripts/select_validation_policies.py \
  --output-root artifacts/debug/024_fakevlm_prefill_global_pareto
python artifacts/debug/024_fakevlm_prefill_global_pareto/scripts/validate_pareto_speed.py \
  --output-root artifacts/debug/024_fakevlm_prefill_global_pareto \
  --batch-size 16
python artifacts/debug/024_fakevlm_prefill_global_pareto/scripts/validate_policy_quality.py \
  --output-root artifacts/debug/024_fakevlm_prefill_global_pareto \
  --policies validation

# 8. Summarize.
MPLCONFIGDIR=/tmp/matplotlib-024 python artifacts/debug/024_fakevlm_prefill_global_pareto/scripts/summarize_validation.py \
  --output-root artifacts/debug/024_fakevlm_prefill_global_pareto
```

Use `--max-modules`, `--sample-limit`, `--max-policies`, and `--points` for smoke runs.
