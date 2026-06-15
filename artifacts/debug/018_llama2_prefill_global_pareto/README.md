# Llama2 Prefill Global-Coeff Pareto

This debug run rebuilds the 008 prefill-only Pareto workflow with the 017-style multiplicative proxy:

`quality_cost(module, method) = local_error * global_coef * layer_coef[layer] * type_coef[type]`

## Current State

- Candidate methods: `dense_bf16`, `dense_nvfp4`, `sparse_bf16`, `sparse_nvfp4`.
- Marlin W4A16 is not used as an optimized Pareto candidate by default because no trusted Marlin-specific per-module global-coeff proxy is available in this run.
- Marlin is still included as a real uniform baseline when existing single-method measurements are summarized.
- Fresh prefill latency is copied from 008 to keep the scenario aligned with the earlier prefill-only experiment.

## Generated Outputs

- `global_coefficients/proxy_ablation_coefficients.json`
- `costs/module_method_candidates.csv`
- `pareto/pareto_unique_points.csv`
- `validation/selected_pareto_points.csv`
- `summary/analysis.md`
- `summary/prefill_only_comparison.csv`
- `plots/*.png`
- `showcase/showcase_summary.md`
- `showcase/speed_vs_nll_showcase.png`
- `showcase/speed_vs_arc_showcase.png`
- `showcase/method_counts_showcase.png`

## Presentation Showcase

The full 29-point frontier is retained for auditability, but the presentation view should use the compact showcase:

```bash
MPLCONFIGDIR=/tmp/matplotlib-018 python artifacts/debug/018_llama2_prefill_global_pareto/scripts/build_showcase_outputs.py \
  --output-root artifacts/debug/018_llama2_prefill_global_pareto
```

Default selected Pareto points:

- `P000`: dense reference.
- `P020`: conservative quality-preserving point, 1.24x speedup with ARC unchanged vs dense.
- `P024`: main favorable point, faster and much lower NLL than uniform sparse baselines.
- `P026`: aggressive favorable point, faster than uniform sparse baselines with lower NLL damage.

`P015` remains in the full validation table, but it is intentionally omitted from the compact showcase because its ARC-Challenge limit-128 score is lower than nearby points despite a normal NLL delta.

## Rebuild Commands

```bash
MPLCONFIGDIR=/tmp/matplotlib-018 python artifacts/debug/017_global_coef_structural_ablation/scripts/fit_proxy_ablation.py \
  --output-root artifacts/debug/018_llama2_prefill_global_pareto \
  --methods sparse_bf16,dense_nvfp4,sparse_nvfp4 \
  --policies-csv-template 'artifacts/debug/018_llama2_prefill_global_pareto/stratified/policies/stratified_policies_{method}.csv' \
  --loss-tag stratified \
  --output-subdir global_coefficients \
  --expected-examples 0 \
  --steps 3000

python artifacts/debug/018_llama2_prefill_global_pareto/scripts/build_cost_table.py \
  --output-root artifacts/debug/018_llama2_prefill_global_pareto \
  --latency-source fresh

python artifacts/debug/018_llama2_prefill_global_pareto/scripts/optimize_pareto.py \
  --output-root artifacts/debug/018_llama2_prefill_global_pareto

python artifacts/debug/018_llama2_prefill_global_pareto/scripts/select_validation_policies.py \
  --output-root artifacts/debug/018_llama2_prefill_global_pareto
```

## Full Local GPU Validation

Run E2E and quality validation separately. The launcher uses one GPU per point and assigns GPUs in `7,6,5,4,3,2` order by default.

```bash
python artifacts/debug/018_llama2_prefill_global_pareto/scripts/launch_local_validation.py \
  --output-root artifacts/debug/018_llama2_prefill_global_pareto \
  --kind e2e \
  --extra-args '--warmup-iters 3 --iters 10'

python artifacts/debug/018_llama2_prefill_global_pareto/scripts/launch_local_validation.py \
  --output-root artifacts/debug/018_llama2_prefill_global_pareto \
  --kind quality \
  --extra-args '--arc-limit 128'

python artifacts/debug/018_llama2_prefill_global_pareto/scripts/summarize_validation.py \
  --output-root artifacts/debug/018_llama2_prefill_global_pareto
```

The validation scripts load real method-specific compressed artifacts from `artifacts/results/main/003_llama2_7b_arc_easy_accuracy/prepared/<method>/model.pt` and install the matching runtime backend. Sparse methods use `prune=False` at runtime conversion because the artifact weights are already compressed.
