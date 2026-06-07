# Qwen3.5-9B Policy Ablation Analysis

## Purpose

This experiment checks whether the manual/pred difference on Qwen3.5-9B `normal_01` can be explained by stable per-linear extra costs.

The tested variants swap the only major policy differences:

- `mlp.down_proj`: manual uses `marlin_nvfp4 -> marlin_nvfp4`; pred uses `dense_nvfp4 -> marlin_nvfp4`.
- `self_attn.k_proj` and `self_attn.v_proj`: manual uses `marlin_nvfp4 -> marlin_nvfp4`; pred uses `dense_bf16 -> dense_bf16`.

## Main Result

Except for `single_sparse_bf16`, all hybrid variants land in a narrow band around `4.02-4.05s` E2E:

| Variant | E2E mean ms | Min/Max ms |
| --- | ---: | ---: |
| `pred_down_kv_to_manual` | 4024.235 | 4009.253 / 4045.904 |
| `manual_down_to_pred` | 4028.484 | 4006.602 / 4045.029 |
| `manual_down_kv_to_pred` | 4031.660 | 4024.519 / 4041.056 |
| `pred` | 4032.904 | 4018.727 / 4043.485 |
| `manual_kv_to_pred` | 4036.561 | 4006.253 / 4061.279 |
| `pred_down_to_manual` | 4038.542 | 4008.598 / 4064.553 |
| `pred_kv_to_manual` | 4048.103 | 4014.532 / 4114.239 |
| `manual` | 4051.349 | 3713.234 / 4394.600 |

The differences between these hybrid variants are mostly within tens of milliseconds. That is close to or smaller than repeat variance in this setup, so these swaps do not expose a stable fixed per-linear correction term.

## sparse_bf16 Is Bimodal

`single_sparse_bf16` was not stable:

```text
4681.5 ms
4681.6 ms
3650.7 ms
```

This confirms that sparse_bf16 full-model timing is affected by global library/runtime warm state. A single E2E sample can make sparse_bf16 look either clearly best or clearly worse.

## Interpretation

There are two separate issues:

1. The old manual module benchmark was incorrect because it repeated sparse_bf16 cold-start costs.
2. Even after fixing cold/steady accounting, per-shape standalone selection is still not a true full-model oracle.

A true oracle must evaluate the complete policy under the real full-model execution path. Per-linear additive selection can fail because:

- sparse/cuSPARSELt cold-start is global by shape/backend, not independent per module;
- backend choices can change full-model runtime state, allocator/cache behavior, and first-token behavior;
- the measured policy differences are smaller than run-to-run variance for several close variants;
- non-linear work and attention/linear interaction are not captured by standalone linear totals.

## Practical Recommendation

For a reliable oracle, use full-model or layer-replay policy search:

1. Warm the runtime globally with all candidate backend/shape pairs before measuring policy variants.
2. Use multiple repeats and compare median/min after warmup, not one sample.
3. Measure complete policy variants, not only independent per-linear choices.
4. If speed is a concern, do local full-model ablation only for uncertain groups after predictor/manual narrows the candidate set.

At this point, a simple fixed extra cost per linear is not well supported by the data. The dominant mismatch is global/runtime-state dependent rather than a clean additive constant.
