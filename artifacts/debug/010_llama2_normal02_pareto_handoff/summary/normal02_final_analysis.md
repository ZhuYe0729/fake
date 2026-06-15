# Llama2 Normal-02 Quality-Speed Pareto Analysis

## Scenario

```text
model = llama2-7b
scenario = normal_02
batch_size = 1
input_tokens = 16384
output_tokens = 256
```

This directory validates a quality-constrained Pareto policy search for the decode-heavy `normal_02` setting.

The optimizer uses:

```text
minimize predicted_total_latency
subject to quality_cost <= budget
```

where the quality cost is the existing Llama2 module-local proxy:

```text
local_rel_mse_log_numel_layer_family
```

The speed source is the warm-E2E-aligned normal_02 candidate table from:

```text
fake/artifacts/results/main/003_llama2_oracle_summary/pred/normal_02/llama2-7b_pred_candidates.csv
```

## Candidate Table

The normal_02 candidate table contains `1344` rows:

```text
224 Llama2 linear modules x 6 candidate methods
```

Supported methods:

```text
dense_bf16
dense_nvfp4
marlin_nvfp4
dense_nvfp4_prefill_marlin_decode
```

Unsupported methods:

```text
sparse_bf16   # M%8 != 0
sparse_nvfp4  # M%32 != 0
```

Sparse methods are therefore excluded from the normal_02 Pareto frontier.

## Stable E2E Result

The original reliable deployment points were `0`, `7`, and `9`; after the OOM fix below, points `4`, `5`, `6`, and `8` also have stable process-per-repeat E2E measurements.

Stable E2E was measured with one fresh Python process per repeat:

```text
repeats = 3
iters_per_process = 1
warmup_iters = 1
gpu = 7
```

| Point | Policy Shape | E2E Mean (ms) | Std (ms) | Speedup vs Dense | NLL Delta | ARC-Challenge acc_norm |
|-------|--------------|---------------|----------|------------------|-----------|------------------------|
| 0 | `224 bf16` | 9026.0 | 4.5 | 1.000x | 0.000000 | 0.4609375 |
| 7 | `153 bf16 + 71 dense_nvfp4->marlin` | 8340.8 | 12.4 | 1.082x | 0.028936 | 0.4609375 |
| 9 | `128 marlin + 96 dense_nvfp4->marlin` | 7394.2 | 5.5 | 1.221x | 0.036796 | 0.4609375 |

The stable E2E ranking matches the predicted latency ranking:

```text
point_9 < point_7 < point_0
```

The main operating point is `point_009`: it gives about `1.22x` stable E2E speedup with a small NLL increase and no ARC-Challenge acc_norm drop at limit 128.

The full measured curve is now:

| Point | Pred Total (ms) | E2E Mean (ms) | Speedup vs Dense | Backend Shape |
|-------|----------------:|--------------:|-----------------:|---------------|
| 0 | 4176.5 | 9026.0 | 1.000x | `224 bf16` |
| 4 | 3957.5 | 8973.6 | 1.006x | `208 bf16 + 16 dense_nvfp4->marlin` |
| 5 | 3800.0 | 8871.1 | 1.017x | `197 bf16 + 27 dense_nvfp4->marlin` |
| 6 | 3562.1 | 8524.9 | 1.059x | `175 bf16 + 5 marlin + 44 dense_nvfp4->marlin` |
| 7 | 3190.3 | 8340.8 | 1.082x | `153 bf16 + 71 dense_nvfp4->marlin` |
| 8 | 2857.2 | 7812.8 | 1.155x | `56 bf16 + 72 marlin + 96 dense_nvfp4->marlin` |
| 9 | 2829.1 | 7394.2 | 1.221x | `128 marlin + 96 dense_nvfp4->marlin` |

## Quality Curve

Quality was evaluated for points:

```text
0, 4, 5, 6, 7, 8, 9
```

| Point | Quality Cost | NLL | NLL Delta | ARC Acc | ARC Acc Norm |
|-------|--------------|-----|-----------|---------|--------------|
| 0 | 0.0000 | 2.039499 | 0.000000 | 0.40625 | 0.4609375 |
| 4 | 0.6481 | 2.064522 | 0.025023 | 0.40625 | 0.46875 |
| 5 | 1.3055 | 2.064560 | 0.025061 | 0.421875 | 0.4609375 |
| 6 | 2.5973 | 2.065575 | 0.026076 | 0.40625 | 0.4609375 |
| 7 | 5.2974 | 2.068435 | 0.028936 | 0.4296875 | 0.4609375 |
| 8 | 10.4523 | 2.072573 | 0.033074 | 0.40625 | 0.4453125 |
| 9 | 16.5301 | 2.076295 | 0.036796 | 0.40625 | 0.4609375 |

Correlation summary:

```text
quality_cost vs NLL:
  Pearson  = 0.6942
  Spearman = 1.0

quality_cost vs ARC-Challenge acc_norm:
  Pearson  = -0.4529
  Spearman = -0.5345
```

Interpretation:

- The quality proxy gives the correct NLL ordering.
- ARC-Challenge limit=128 is too insensitive for fine-grained optimization here.
- ARC should be treated as external validation, not the optimization target.

## OOM Diagnosis And Fix

The earlier points `4`, `5`, `6`, and `8` failed only in the old multi-point single-process validator. They passed policy replacement and then OOMed during benchmark with several GB of PyTorch memory "reserved but unallocated", which points to CUDA allocator fragmentation rather than an inherent 7B model capacity issue.

The fix is to run E2E validation as one fresh Python process per point/repeat and enable:

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

`validate_pareto_e2e.py` now sets the allocator config by default and more aggressively frees benchmark temporaries. `run_stable_e2e_repeats.py` now accepts `--run-name` so different repeat batches do not overwrite each other.

After the fix, points `4`, `5`, `6`, and `8` all run successfully on the same 32GB GPU:

| Point | Repeats | Pred Total (ms) | E2E Mean (ms) | Std (ms) | Speedup vs Dense | Policy Shape |
|-------|---------|----------------:|--------------:|---------:|-----------------:|--------------|
| 4 | 3/3 | 3957.5 | 8973.6 | 4.7 | 1.006x | `208 bf16 + 16 dense_nvfp4->marlin` |
| 5 | 3/3 | 3800.0 | 8871.1 | 11.5 | 1.017x | `197 bf16 + 27 dense_nvfp4->marlin` |
| 6 | 3/3 | 3562.1 | 8524.9 | 30.8 | 1.059x | `175 bf16 + 5 marlin + 44 dense_nvfp4->marlin` |
| 8 | 3/3 | 2857.2 | 7812.8 | 101.3 | 1.155x | `56 bf16 + 72 marlin + 96 dense_nvfp4->marlin` |

The combined stable E2E curve is saved at:

```text
validation/stable_e2e_repeats/stable_e2e_repeats_all_points.csv
```

## Baseline Comparison

Existing `003` normal_02 E2E baselines:

| Baseline | E2E (ms) | Speedup vs Dense |
|----------|----------|------------------|
| all_dense_bf16 | 9101 | 1.000x |
| all_dense_nvfp4 | 17349 | 0.525x |
| all_dense_nvfp4_prefill_marlin_decode | 7762 | 1.173x |
| all_marlin_nvfp4 | 7718 | 1.179x |
| pred_policy | 7282 | 1.250x |
| oracle_policy | 7427 | 1.225x |

The stable `point_009` result is close to the existing pred/oracle region:

```text
point_009 stable: 7394 ms
003 pred policy:  7282 ms
003 oracle:       7427 ms
```

This is consistent with point_009 having the same high-level backend structure as the normal_02 pred/oracle policies:

```text
128 marlin_nvfp4 + 96 dense_nvfp4_prefill_marlin_decode
```

## Conclusions

1. `llama2-7b normal_02` is a positive case for quality-constrained speed optimization.
2. The predicted latency ranking matches stable real E2E ranking for the deployable points.
3. The quality proxy is reliable for NLL ordering.
4. `point_009` is the recommended normal_02 operating point on this setup.
5. ARC-Challenge limit=128 is not sensitive enough to distinguish the selected points.
6. Future E2E timing for this scenario should use process-per-repeat measurement to avoid long-process CUDA fragmentation.

## Recommended Next Step

Before expanding to Qwen, run the same normal_02 workflow on `llama3.1-8b`:

```text
candidate table -> Pareto frontier -> stable E2E for 0/mid/fast -> NLL + task validation
```

If Llama3.1 shows the same behavior, the method story becomes much stronger because the result is no longer Llama2-specific.
