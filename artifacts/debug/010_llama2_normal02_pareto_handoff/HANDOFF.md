# 010 Handoff: Llama2 Normal-02 Pareto Smoke

## Purpose

This handoff replaces the immediate `normal_01` follow-up with a `normal_02` first pass.

The `009` smoke showed that `normal_01` (`output_tokens=32`) is awkward for this study: prefill prediction remains good, but total E2E ranking is sensitive to short-decode overhead and mixed-backend dispatch effects. For the next reviewable step, focus on `normal_02`, where decode length is long enough for decode-oriented methods such as Marlin to matter more clearly.

## Scope

Create a small, reviewable Pareto smoke package for:

```text
model = llama2-7b
scenario = normal_02
batch_size = 1
input_tokens = 16384
output_tokens = 256
```

Use this new isolated root:

```text
fake/artifacts/debug/010_llama2_normal02_pareto_handoff
```

Recommended subdirectories:

```text
scripts/
costs/
pareto/
validation/
summary/
```

Do not modify the main framework. Do not modify the `008` or `009` result package except to read from them.

## Non-Goals

Do not do these in this step:

- Do not run `normal_01` again.
- Do not run `prefill_only` again.
- Do not run llama3.1-8B.
- Do not run Qwen3.5-9B.
- Do not run full NLL or ARC validation.
- Do not produce paper-level plots.
- Do not run all Pareto points through E2E unless the 3-4 point smoke is already complete and explicitly requested later.

Stop after the normal_02 smoke package is complete.

## Context From Previous Steps

### 008: Prefill-Only Worked

Root:

```text
fake/artifacts/debug/008_llama2_pareto_quality_speed
```

Important result:

- Per-linear latency strongly matched real prefill E2E ranking.
- Quality proxy strongly matched NLL ranking.
- This validates the constrained optimization form:

```text
minimize latency
subject to quality_cost <= budget
```

### 009: Normal-01 Exposed Decode Modeling Issues

Root:

```text
fake/artifacts/debug/009_llama2_normal01_pareto_handoff
```

Important result:

- `normal_01` prefill ranking was still good.
- Total E2E ranking was poor because decode length was short and mixed-backend overhead dominated.
- Do not directly copy the `009` result interpretation to `normal_02`.
- You may copy and adapt the `009/scripts`, but fix the statistics issue described below.

Important bug to fix when copying `009`:

`validate_pareto_e2e.py` in `009` records `e2e_times_ms` for all iterations but reports prefill/decode/total from the final iteration. In `010`, report mean/median/min/max properly.

## Existing Normal-02 Data To Reuse

Prefer these existing warm-E2E-aligned sources:

```text
fake/artifacts/results/main/003_llama2_oracle_summary
```

Useful files:

```text
pred/normal_02/llama2-7b_pred_candidates.csv
pred/normal_02/llama2-7b_policy.json
pred/normal_02/llama2-7b_full_e2e.csv
oracle/normal_02/llama2-7b_policy.json
oracle/normal_02/llama2-7b_full_e2e.csv
single/dense_bf16/normal_02/llama2-7b_full_e2e.csv
single/dense_nvfp4/normal_02/llama2-7b_full_e2e.csv
single/dense_nvfp4_prefill_marlin_decode/normal_02/llama2-7b_full_e2e.csv
single/marlin_nvfp4/normal_02/llama2-7b_full_e2e.csv
single/sparse_bf16/normal_02/llama2-7b_full_e2e.csv
single/sparse_nvfp4/normal_02/llama2-7b_full_e2e.csv
```

Known normal_02 E2E sanity values from `003`:

```text
all_dense_bf16:                         e2e ~= 9101 ms
all_dense_nvfp4:                        e2e ~= 17349 ms
all_dense_nvfp4_prefill_marlin_decode:  e2e ~= 7762 ms
all_marlin_nvfp4:                       e2e ~= 7718 ms
all_sparse_bf16:                        e2e ~= 10335 ms
all_sparse_nvfp4:                       e2e ~= 21729 ms
pred policy:                            e2e ~= 7282 ms
oracle policy:                          e2e ~= 7427 ms
```

Interpretation:

- Pure `dense_nvfp4` and `sparse_nvfp4` are bad for decode-heavy normal_02.
- Marlin decode is important.
- `dense_nvfp4_prefill_marlin_decode` and `marlin_nvfp4` are expected to be useful.
- Sparse methods should be included only if the latency source marks the actual strategy as supported.

## Methods And Candidate Rules

Candidate methods to consider:

```text
dense_bf16
dense_nvfp4
sparse_bf16
sparse_nvfp4
marlin_nvfp4
dense_nvfp4_prefill_marlin_decode
```

However, do not blindly force all 6 methods per module if the normal_02 latency source says a method is unsupported.

Preferred speed source:

```text
fake/artifacts/results/main/003_llama2_oracle_summary/pred/normal_02/llama2-7b_pred_candidates.csv
```

That file already has columns like:

```text
candidate
prefill_backend
decode_backend
supported
reason
prefill_ms
decode_ms
total_ms
weighted_total_ms
online_conversion_ms
linear_group
n
k
count
```

For each module, map by:

```text
linear_group, out_features(n), in_features(k)
```

Use only `supported=True` rows unless you have a concrete framework reason to override.

Important:

- `dense_nvfp4_prefill_marlin_decode` should reuse the dense NVFP4 quality cost.
- `marlin_nvfp4` should also reuse dense NVFP4 quality cost.
- Sparse methods should reuse their own sparse quality costs.
- Do not use ARC accuracy as the optimization objective.

## Quality Source

Reuse the Llama2 quality proxy from `007`:

```text
fake/artifacts/debug/007_llama2_quality_modeling/sensitivity/module_method_errors.csv
```

Use the same quality formula unless a real bug is found:

```text
local_rel_mse_log_numel_layer_family
```

This step is about speed/modeling smoke. Do not run full NLL/ARC yet.

## Optimization Formulation

Keep the same constrained form:

```text
minimize predicted_total_latency_ms
subject to total_quality_cost <= budget
```

For normal_02:

```text
predicted_total_latency_ms =
    sum(prefill_ms)
  + output_tokens * sum(decode_ms)
  + sum(conversion_ms)
```

If using `llama2-7b_pred_candidates.csv`, its `total_ms` should already correspond to:

```text
prefill_ms + 256 * decode_ms + online_conversion_ms
```

Verify this numerically in the script and write the check result to metadata.

Use a budget grid similar to:

```text
0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0
```

The exact grid may be adjusted if it produces too few unique points.

## Required Outputs

### Candidate Table

Create:

```text
costs/module_method_candidates.csv
costs/build_cost_table_metadata.json
```

Required columns:

```text
module_name
layer
module_family
module_type
method
prefill_backend
decode_backend
supported
unsupported_reason
quality_cost
prefill_ms
decode_ms
conversion_ms
total_ms
latency_cost
latency_gain_vs_dense
quality_delta_vs_dense
quality_formula
latency_source
```

Acceptance criteria:

- Every Llama2 linear module has a `dense_bf16` candidate.
- Missing latency rows are reported explicitly.
- Unsupported candidates are either excluded from optimization or retained with `supported=False` and never selected.
- The metadata records whether `total_ms == prefill_ms + 256 * decode_ms + conversion_ms`.

### Pareto Frontier

Create:

```text
pareto/pareto_points.csv
pareto/pareto_unique_points.csv
pareto/policies/*.json
summary/frontier_summary.csv
summary/method_cost_summary.csv
```

Acceptance criteria:

- At least 5 unique Pareto points.
- Point 0 should be all/effectively all `dense_bf16`.
- Fastest point should be faster than dense in predicted latency.
- Frontier should be interpretable; expected useful methods are `marlin_nvfp4` and/or `dense_nvfp4_prefill_marlin_decode`.
- If sparse methods are selected, explain exactly why they are supported and why the optimizer chose them.

### E2E Smoke Validation

Validate 3-4 points with real full-model E2E:

```text
point_000
one conservative/middle point
one aggressive point
fastest point
```

Use a high-numbered free GPU. Prefer GPU7, then GPU6, etc. Keep GPU0 and GPU1 free.

Use:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=7 \
python fake/artifacts/debug/010_llama2_normal02_pareto_handoff/scripts/validate_pareto_e2e.py \
  --gpu 7 \
  --points 0,<middle>,<aggressive>,<fastest> \
  --warmup-iters 1 \
  --iters 3
```

The benchmark must run real full-model inference with real kernels.

Output:

```text
validation/pareto_e2e_validation.csv
validation/pareto_e2e_validation_metadata.json
validation/validation_correlations.csv
summary/normal02_smoke_comparison.csv
summary/normal02_smoke_summary.md
```

Required E2E columns:

```text
point_index
predicted_total_latency_ms
predicted_prefill_latency_ms
predicted_decode_latency_ms
predicted_conversion_latency_ms
e2e_total_mean_ms
e2e_total_median_ms
e2e_total_min_ms
e2e_total_max_ms
e2e_prefill_mean_ms
e2e_decode_avg_mean_ms
e2e_decode_first_mean_ms
e2e_decode_steady_mean_ms
e2e_times_ms
replaced_linear_count
skipped_linear_count
backend_counts
requested_gpu
local_gpu
iters
warmup_iters
```

Acceptance criteria:

- `replaced_linear_count == 224`.
- `skipped_linear_count == 0`.
- Real E2E ranking should broadly follow predicted total latency. If it does not, record it honestly.
- Compare smoke points against the existing `003` single/pred/oracle normal_02 E2E baselines.

## Minimal Summary Requirements

Write:

```text
summary/normal02_smoke_summary.md
summary/normal02_smoke_comparison.csv
```

The Markdown must answer:

1. Did the candidate table use supported normal_02 strategies?
2. How many unique Pareto points were produced?
3. Which methods appear on the frontier?
4. Did `dense_nvfp4_prefill_marlin_decode` appear and was it selected?
5. Did `marlin_nvfp4` appear and was it selected?
6. Were sparse methods selected? If yes, why are they valid for normal_02?
7. Did predicted latency ranking match real E2E ranking for the smoke points?
8. How do the smoke points compare to existing all-dense, all-marlin, all-hybrid, pred, and oracle baselines?
9. Should the next step be quality validation or latency model correction?

The CSV should include:

```text
row_type
label
point_index
quality_cost
predicted_total_latency_ms
predicted_prefill_latency_ms
predicted_decode_latency_ms
predicted_conversion_latency_ms
e2e_total_mean_ms
e2e_speedup_vs_dense
backend_counts
notes
```

## Review Checklist To Report Back

When done, report exactly:

1. Path to the `010` directory.
2. Candidate rows count.
3. Missing/unsupported latency count.
4. Number of unique Pareto points.
5. Method counts for each E2E-validated point.
6. Which 3-4 points were E2E validated.
7. Predicted total latency vs real E2E mean for those points.
8. Whether E2E ranking matches predicted ranking.
9. Whether `dense_nvfp4_prefill_marlin_decode` was selected.
10. Whether `marlin_nvfp4` was selected.
11. Whether sparse methods were selected and whether they were actually supported.
12. Any OOM, GPU, or unsupported-kernel failures.

## Stop Condition

Stop after the `normal_02` 3-4 point smoke package is complete.

Do not proceed to quality validation, all-point E2E, `normal_01`, llama3.1, or Qwen until this result is reviewed.

