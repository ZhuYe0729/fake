# Llama2 normal_02 Warm E2E Gap Trace

## Purpose

This debug run checks where standalone manual module measurements disagree with actual full-model execution for `llama2-7b normal_02`.

The trace applies the real `manual` and `pred` policies from `002_warm_e2e_aligned_policy_retest`, hooks the replaced linear modules inside the full model, and records CUDA event timings for:

- full prefill;
- first decode step;
- eight steady decode steps.

## Files

- `scripts/trace_llama2_normal02_linear_in_model.py`: full-model hook trace script.
- `results/manual_raw_linear_trace.csv`, `results/pred_raw_linear_trace.csv`: per-module call traces.
- `results/manual_step_group_projection.csv`, `results/pred_step_group_projection.csv`: per-group prefill and decode-step aggregation.
- `results/pred_vs_manual_step_group_delta.csv`: full-model trace delta.
- `results/standalone_vs_in_model_key_groups.csv`: key comparison against standalone manual candidates.

## Main Result

The disagreement is concrete: standalone manual scoring says the pred-style candidate is slower for several groups, while full-model hook trace shows it is faster.

| group | standalone pred - manual ms | in-model pred - manual ms |
|---|---:|---:|
| `mlp.down_proj` | +69.50 | -12.89 |
| `mlp.gate_proj` | +181.23 | -98.20 |
| `self_attn.o_proj` | +2.70 | -152.98 |
| `self_attn.q_proj` | +91.57 | -81.19 |

Negative in-model delta means pred is faster inside the real model.

## Concrete Mismatch

- For `mlp.gate_proj`, standalone manual over-penalizes `dense_nvfp4->marlin_nvfp4`: it predicts this candidate is 181 ms worse than all-Marlin, but full-model trace shows it is 98 ms faster. The in-model trace shows the MLP prefill saving dominates, while steady decode per group is essentially the same.
- For `mlp.down_proj`, standalone manual predicts the hybrid candidate is 70 ms worse, but full-model trace shows it is about 13 ms faster.
- For `self_attn.o_proj/q_proj`, standalone manual picks dense bf16, but in-model trace shows Marlin is faster for both groups.

## Caveat

Forward hooks add event-recording overhead, so absolute totals are for diagnosis rather than final benchmark numbers. The useful signal is the group-level relative direction under the same trace method.
