# Next Agent Guide: Llama2 Pareto Quality-Speed Follow-up

## Current State

This workspace implements and validates a first-pass quality-constrained Pareto optimizer for `llama2-7b` under `prefill_only`.

Root directory:

```bash
fake/artifacts/debug/008_llama2_pareto_quality_speed
```

Scenario:

```text
model = llama2-7b
scenario = prefill_only
batch_size = 16
input_tokens = 1024
output_tokens = 0
```

What is already complete:

- Fresh real-kernel group microbench latency was collected on physical GPU7.
- Candidate table was built for `224 linear modules x 5 methods = 1120 rows`.
- A constrained optimizer generated 11 unique Pareto points.
- All 11 Pareto points were validated for NLL + ARC-Challenge limit=128.
- 5 representative Pareto points were validated with real full-model prefill-only E2E latency.
- Policy explanation tables were generated.

Key existing results:

- Predicted linear latency vs real E2E latency on selected 5 points:
  - Pearson `0.9997408220927038`
  - Spearman `1.0`
- Quality cost vs NLL on all 11 quality points:
  - Pearson `0.9691908137987784`
  - Spearman `1.0`
- Quality cost vs ARC-Challenge acc_norm on all 11 quality points:
  - Pearson `-0.9652633710759109`
  - Spearman `-0.789776981742655`

Important outputs:

```bash
latency/prefill_latency.csv
costs/module_method_candidates.csv
pareto/pareto_unique_points.csv
validation/pareto_quality_validation.csv
validation/pareto_e2e_validation.csv
validation/pareto_validation_joined.csv
validation/validation_correlations.csv
summary/policy_transition_summary.csv
summary/analysis.md
```

## Goal For The Next Agent

Produce a complete, presentation-ready `llama2-7b + prefill_only` result package:

1. Run real full-model E2E latency for all 11 unique Pareto points.
2. Integrate uniform baselines into the same summary:
   - all dense_bf16
   - all dense_nvfp4
   - all sparse_bf16
   - all sparse_nvfp4
   - optionally all marlin_nvfp4, but it is expected to be poor for prefill-only
3. Generate first-pass plots and an analysis Markdown that clearly supports the Pareto story.

Do not expand to `normal_01`, `normal_02`, llama3.1, or Qwen until the prefill-only result package is complete.

## Step 1: Run E2E For All Pareto Points

Current E2E validation covers points `0,2,5,8,10`. Run all 11 points:

```bash
cd /root/wja/project/my/cospaq

CUDA_VISIBLE_DEVICES=7 python fake/artifacts/debug/008_llama2_pareto_quality_speed/scripts/validate_pareto_e2e.py \
  --gpu 7 \
  --points 0,1,2,3,4,5,6,7,8,9,10 \
  --warmup-iters 1 \
  --iters 3
```

If GPU7 is occupied, use another free high-numbered GPU and set `CUDA_VISIBLE_DEVICES` accordingly. Keep GPU0 and GPU1 free unless explicitly told otherwise.

After it finishes:

```bash
python fake/artifacts/debug/008_llama2_pareto_quality_speed/scripts/summarize_validation.py
```

Expected outputs:

```bash
validation/pareto_e2e_validation.csv          # 12 lines: header + 11 rows
validation/pareto_validation_joined.csv       # 12 lines if all E2E points join with quality rows
validation/validation_correlations.csv
```

Acceptance criteria:

- `replaced_linear_count` is `224` for every row.
- `skipped_linear_count` is `0` for every row.
- E2E latency should broadly decrease as budget increases.
- `predicted_linear_latency_ms` vs `e2e_prefill_mean_ms` should keep Spearman close to `1.0`.

## Step 2: Add Uniform Baselines To The Same Table

Need a single comparison table containing both:

- Pareto points
- uniform single-method policies

Uniform policies to include:

```text
uniform_dense_bf16
uniform_dense_nvfp4
uniform_sparse_bf16
uniform_sparse_nvfp4
uniform_marlin_nvfp4 optional
```

Use existing data where possible:

- Method quality totals are already summarized in:

```bash
summary/method_cost_summary.csv
```

- Existing all-method quality baselines can be cross-checked from:

```bash
fake/artifacts/debug/007_llama2_quality_modeling/arc_challenge_limit128/ablations/policy_quality_results.csv
```

Relevant all-policy rows from earlier runs:

```text
dense_bf16 none
dense_nvfp4 all
sparse_bf16 all
sparse_nvfp4 all
```

For E2E uniform baselines, either:

1. Reuse existing full-model E2E data from:

```bash
fake/artifacts/results/main/003_llama2_oracle_summary/single/*/prefill_only/llama2-7b_full_e2e.csv
```

2. Or run them through the same E2E validation path by creating uniform policy JSONs in 008.

Recommended: reuse existing data first, then only rerun if the timing mode is incompatible.

Create a new script, recommended name:

```bash
fake/artifacts/debug/008_llama2_pareto_quality_speed/scripts/build_baseline_comparison.py
```

It should write:

```bash
summary/prefill_only_comparison.csv
```

Suggested columns:

```text
row_type                  # pareto or uniform
label                     # point_000, all_dense_nvfp4, etc.
point_index
quality_cost
predicted_linear_latency_ms
e2e_prefill_mean_ms
e2e_speedup_vs_dense
nll
nll_delta_vs_dense
arc_acc
arc_acc_norm
backend_counts
source
```

Acceptance criteria:

- The dense baseline is shared consistently.
- Pareto points are not worse than obvious uniform baselines at comparable quality cost.
- If a uniform baseline is missing a metric, mark it blank and include a `source` explanation.

## Step 3: Generate Plots

Create a plotting script:

```bash
fake/artifacts/debug/008_llama2_pareto_quality_speed/scripts/plot_prefill_pareto.py
```

Use `matplotlib`. Output PNG and PDF if possible.

Recommended plots:

1. Speed vs NLL:

```text
x = e2e_speedup_vs_dense
y = nll_delta_vs_dense
highlight Pareto points and uniform baselines separately
```

2. Speed vs ARC-Challenge:

```text
x = e2e_speedup_vs_dense
y = arc_acc_norm
```

3. Method counts along frontier:

```text
x = point_index or quality_cost
y = stacked counts for dense_bf16, dense_nvfp4, sparse_bf16, sparse_nvfp4, marlin_nvfp4
```

4. Predicted vs real latency:

```text
x = predicted_linear_latency_ms
y = e2e_prefill_mean_ms
```

Expected output directory:

```bash
plots/
```

Suggested filenames:

```bash
plots/speed_vs_nll.png
plots/speed_vs_arc_challenge.png
plots/method_counts_frontier.png
plots/predicted_vs_e2e_latency.png
```

Acceptance criteria:

- Plots clearly distinguish Pareto points from uniform baselines.
- Axes and legends are readable.
- The figure should be usable directly in an internal report.

## Step 4: Write Final Analysis Markdown

Create:

```bash
summary/prefill_only_final_analysis.md
```

It should include:

1. A concise claim:

```text
Constrained per-linear assignment finds useful speed-quality tradeoffs under prefill-only.
```

2. Evidence:

- E2E speedup of selected Pareto points.
- NLL and ARC-Challenge trend.
- Correlation between predicted linear latency and real E2E.
- Comparison against uniform baselines.

3. Interpretation:

- Conservative budgets select low-risk MLP `dense_nvfp4`.
- Medium budgets introduce `sparse_bf16`.
- Aggressive budgets use `sparse_nvfp4` for selected MLP-heavy choices.
- `marlin_nvfp4` is not useful for prefill-only because measured prefill latency is worse than dense.

4. Limitations:

- Only `llama2-7b`.
- Only `prefill_only`.
- Quality proxy is validated against NLL and ARC-Challenge limit=128, not full downstream suite.
- E2E timing has only 3 iterations per point.

5. Next research steps:

- Extend to `normal_01`.
- Extend to `normal_02`.
- Add decode and hybrid conversion costs.
- Repeat on llama3.1-8B and Qwen3.5-9B after the scenario logic is stable.

## Important Implementation Notes

### E2E vs Quality Validation Use Different Replacement Paths

Quality validation:

```bash
scripts/validate_pareto_quality.py
```

This loads prepared compressed artifacts and evaluates NLL/ARC. Use it for accuracy/quality.

E2E validation:

```bash
scripts/validate_pareto_e2e.py
```

This converts Pareto policies to `offline_hybrid_policy_v1` and applies real runtime kernels. Use it for speed.

Do not mix these meanings. The E2E script validates kernel speed, not calibrated compressed artifact quality.

### Current Candidate Methods

```text
dense_bf16
dense_nvfp4
sparse_bf16
sparse_nvfp4
marlin_nvfp4
```

For prefill-only, `marlin_nvfp4` is expected to be ignored by the optimizer because its measured prefill latency is poor.

### Quality Formula

Current default:

```text
quality_cost = local_rel_mse * log1p(numel) * layer_weight * family_weight
```

Formula comparison output:

```bash
quality/formula_correlation.csv
```

Do not change the formula in the next step unless the task explicitly becomes formula search. The immediate priority is packaging the current result.

### GPU Usage

Preferred order: GPU7, then GPU6, then GPU5, etc. Avoid GPU0 and GPU1.

Check occupancy:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
```

Use:

```bash
CUDA_VISIBLE_DEVICES=7 ... --gpu 7
```

The scripts handle local CUDA ordinal mapping.

## Final Acceptance Checklist

Before handing back:

- `pareto_e2e_validation.csv` has 11 rows plus header.
- `pareto_quality_validation.csv` has 11 rows plus header.
- `prefill_only_comparison.csv` exists and includes Pareto + uniform baselines.
- At least four plot files exist under `plots/`.
- `summary/prefill_only_final_analysis.md` exists.
- `python -m py_compile fake/artifacts/debug/008_llama2_pareto_quality_speed/scripts/*.py` passes.
- No validation process remains running:

```bash
pgrep -af '008_llama2_pareto_quality_speed/scripts/(validate_pareto_quality|validate_pareto_e2e|bench_prefill_latency)'
```

