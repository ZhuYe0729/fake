# 009 Handoff: Llama2 Normal-01 Pareto Smoke

## Purpose

This handoff is for the next agent to continue the quality-speed Pareto work in a small, reviewable step.

The previous `008` experiment established that the constrained optimization approach works for `llama2-7b` under `prefill_only`. The next step is **not** to expand to every model or every scenario. The next step is to adapt the pipeline to a realistic prefill+decode setting and run a minimal smoke validation.

## Current Baseline To Reuse

Previous root:

```text
fake/artifacts/debug/008_llama2_pareto_quality_speed
```

Important completed files:

```text
summary/prefill_only_comparison.csv
validation/validation_correlations.csv
pareto/pareto_unique_points.csv
costs/module_method_candidates.csv
quality/formula_correlation.csv
plots/*.png
```

Key result from `008`:

- Predicted linear latency vs real E2E prefill latency is very strong:
  - Pearson around `0.9995`
  - Spearman `1.0`
- Quality proxy vs NLL is strong:
  - Pearson around `0.969`
  - Spearman `1.0`
- ARC-Challenge is directionally useful but noisy:
  - Spearman around `-0.79`

Interpretation:

The current method is good enough to move from `prefill_only` to a more realistic scenario. Do not redesign the whole quality proxy in this step.

## Scope For This Step

Create and validate a first `normal_01` smoke pipeline for:

```text
model = llama2-7b
scenario = normal_01
batch_size = 1
input_tokens = 16384
output_tokens = 32
```

The objective is to prove that the optimizer, policy conversion, and real E2E timing logic work when both prefill and decode matter.

## Explicit Non-Goals

Do not do these in this step:

- Do not run llama3.1-8B.
- Do not run Qwen3.5-9B.
- Do not run `normal_02`.
- Do not produce paper-level plots.
- Do not run a full 11-point Pareto quality validation yet.
- Do not change the main framework.
- Do not put experimental scripts outside `fake/artifacts/debug/009_llama2_normal01_pareto_handoff`.

This step should produce a small result package that can be reviewed before continuing.

## Required Output Directory

Use this directory as the new experimental root:

```text
fake/artifacts/debug/009_llama2_normal01_pareto_handoff
```

Recommended subdirectories:

```text
scripts/
latency/
costs/
pareto/
validation/
summary/
```

It is fine to copy scripts from `008` and modify them locally. Do not modify the `008` result package unless there is a clear bugfix and it is documented.

## Methods To Support

Use the same methods as `008`:

```text
dense_bf16
dense_nvfp4
sparse_bf16
sparse_nvfp4
marlin_nvfp4
```

Also include a hybrid candidate for this scenario:

```text
dense_nvfp4_prefill_marlin_decode
```

Important:

- `dense_nvfp4_prefill_marlin_decode` should reuse dense NVFP4 quality cost.
- It should not be treated as identical to pure dense NVFP4 for latency.
- Its total latency must include prefill latency, decode latency, and any required conversion/transition cost if the framework exposes it.
- If conversion cost is not available yet, record it as `0.0` and clearly mark that assumption in metadata.

## Optimization Formulation

Keep the constrained optimization structure:

```text
minimize total_latency
subject to total_quality_cost <= budget
```

For `normal_01`, total latency should be modeled as:

```text
total_latency_ms =
    prefill_latency_ms
  + output_tokens * decode_latency_ms
  + conversion_latency_ms
```

For this step:

- Reuse the `008` module quality proxy.
- Reuse the same quality formula unless there is a concrete bug:

```text
local_rel_mse_log_numel_layer_family
```

Do not use ARC accuracy as the optimization objective.

## Implementation Plan

### Step 1: Copy And Generalize The Core Scripts

Start from these `008` scripts:

```text
scripts/common_pareto.py
scripts/build_cost_table.py
scripts/optimize_pareto.py
scripts/summarize_pareto.py
scripts/validate_pareto_e2e.py
scripts/summarize_validation.py
```

Create local `009/scripts/*` versions.

Required changes:

- Set `DEBUG_ROOT` to the `009` directory.
- Set scenario to:

```python
SCENARIO = {
    "name": "normal_01",
    "batch_size": 1,
    "input_tokens": 16384,
    "output_tokens": 32,
    "m_prefill": 16384,
    "m_decode": 1,
}
```

- Add fields for:

```text
prefill_ms
decode_ms
conversion_ms
total_ms
latency_cost
```

The optimizer should use `latency_cost = total_ms`.

### Step 2: Build A Normal-01 Candidate Table

Create:

```text
costs/module_method_candidates.csv
```

Expected granularity:

```text
224 llama2 linear modules x supported method/candidate choices
```

Required columns:

```text
module_name
layer
module_family
module_type
method
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

Use real-kernel latency sources. If an existing latency summary already has prefill/decode data for `normal_01`, use it and record the exact source path. If not, add a small local benchmark script in `009/scripts`.

Acceptance criteria:

- Every Llama2 linear module has a `dense_bf16` row.
- No selected Pareto policy may contain unsupported kernels.
- `dense_nvfp4_prefill_marlin_decode` appears as a candidate if the framework can express it.
- `marlin_nvfp4` should not be selected for prefill-heavy paths unless its total latency justifies it.

### Step 3: Generate A Small Pareto Frontier

Run the optimizer with a small but useful budget grid, for example:

```text
0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0
```

Output:

```text
pareto/pareto_points.csv
pareto/pareto_unique_points.csv
pareto/policies/*.json
summary/policy_transition_summary.csv
```

Acceptance criteria:

- At least 5 unique Pareto points are produced.
- Point 0 should be all or effectively all `dense_bf16`.
- The fastest point should have lower predicted total latency than dense.
- The method counts should show an interpretable progression from conservative to aggressive compression.

### Step 4: Real E2E Smoke Validation

Validate only 3 Pareto points first:

```text
point_000
one middle point
fastest point
```

Use a high-numbered free GPU, preferably GPU7, then GPU6, etc. Keep GPU0 and GPU1 free.

Suggested command style:

```bash
CUDA_VISIBLE_DEVICES=7 python fake/artifacts/debug/009_llama2_normal01_pareto_handoff/scripts/validate_pareto_e2e.py \
  --gpu 7 \
  --points 0,<middle>,<fastest> \
  --warmup-iters 1 \
  --iters 3
```

The E2E benchmark must run real inference with real kernels, not just packed tensors or synthetic linear calls.

Output:

```text
validation/pareto_e2e_validation.csv
validation/pareto_e2e_validation_metadata.json
```

Required columns:

```text
point_index
predicted_total_latency_ms
predicted_prefill_latency_ms
predicted_decode_latency_ms
predicted_conversion_latency_ms
e2e_mean_ms
e2e_median_ms
e2e_min_ms
e2e_max_ms
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

- `replaced_linear_count` should be `224`.
- `skipped_linear_count` should be `0`.
- Real E2E latency should broadly follow predicted total latency across the 3 points.
- If the fastest predicted point is not fastest in E2E, do not hide it. Record it and explain the likely cause.

### Step 5: Minimal Summary

Write:

```text
summary/normal01_smoke_summary.md
summary/normal01_smoke_comparison.csv
```

The Markdown should answer:

1. Did script generalization work?
2. Did the optimizer produce a sensible frontier?
3. Did real E2E timing follow predicted total latency?
4. Did hybrid `dense_nvfp4_prefill_marlin_decode` appear, and was it useful?
5. What should be reviewed before running a larger validation?

The CSV should include at least:

```text
row_type
label
point_index
quality_cost
predicted_total_latency_ms
predicted_prefill_latency_ms
predicted_decode_latency_ms
predicted_conversion_latency_ms
e2e_mean_ms
e2e_speedup_vs_dense
backend_counts
notes
```

## Quality Validation In This Step

Do not run full NLL or ARC validation unless the E2E smoke already looks reasonable.

If time allows, run NLL only for the same 3 points:

```text
point_000
middle point
fastest point
```

ARC-Challenge can wait for the next iteration.

## Review Checklist For Returning To The Main Agent

When this step is complete, report these exact items:

1. Path to the `009` directory.
2. Number of candidate rows.
3. Number of unique Pareto points.
4. Which 3 points were E2E validated.
5. Whether `replaced_linear_count == 224` and `skipped_linear_count == 0`.
6. Predicted total latency and real E2E latency for those 3 points.
7. Whether the E2E ranking matches predicted ranking.
8. Whether `dense_nvfp4_prefill_marlin_decode` was available and whether it was selected.
9. Any assumptions about conversion latency.
10. Any failed commands or GPU/OOM issues.

## Stop Condition

Stop after the 3-point `normal_01` smoke package is complete.

Do not proceed to:

- full 11-point E2E validation,
- full NLL/ARC validation,
- `normal_02`,
- llama3.1-8B,
- Qwen3.5-9B.

Those should be decided after reviewing this smoke result.

