# Qwen3.5-9B normal_01 Linear/E2E Gap Analysis

## Question

Why does the per-shape manual oracle not reliably win full-model E2E on Qwen3.5-9B `normal_01`?

Scenario:

- `batch_size=1`
- `input_tokens=16384`
- `output_tokens=32`

## Key Finding

The current manual "oracle" is not measuring the same behavior as the real full model for Qwen3.5-9B. In particular, the standalone candidate benchmark severely overestimates `sparse_bf16` decode latency for small `m=1`.

That makes `sparse_bf16` look unusable for normal decoding in `manual_candidates.csv`, even though the real model runs `sparse_bf16` decode at roughly 0.1-0.2 ms per linear call.

## Evidence

### Full-model traced sparse_bf16

`results_sparse_shape/linear_calls.csv` confirms the real model uses:

- prefill input shape: `1x16384xK`
- decode input shape: `1x1xK`

Full-model sparse_bf16 traced linear totals:

- prefill linear sum: about `505 ms`
- decode linear sum: about `944 ms`
- total compressible linear sum: about `1449.6 ms`

Selected real model per-call decode examples:

| Group | Calls | Total ms | Avg ms |
| --- | ---: | ---: | ---: |
| `mlp.gate_proj` | 1024 | 118.471 | 0.1157 |
| `mlp.down_proj` | 1024 | 200.782 | 0.1961 |
| `linear_attn.in_proj_a` | 768 | 79.030 | 0.1029 |

### Current manual candidate benchmark

The current `benchmark_manual_candidate()` reports much slower sparse_bf16 decode:

| Group | standalone sparse_bf16 decode ms |
| --- | ---: |
| `mlp.gate_proj` | about `3.9` |
| `mlp.down_proj` | about `3.9` |
| `linear_attn.in_proj_a` | about `4.6` |

This is 20x-40x slower than the real full-model traced decode calls. That alone is enough to make manual selection avoid `sparse_bf16` even when full E2E can benefit from it.

### Same module is shape/state sensitive

A direct module test showed the same sparse_bf16 module can behave very differently for small decode-like calls:

- `(1, K)` can be several ms.
- `(1, 1, K)` can be about `0.1-0.2 ms` in some sequences.

The complete Qwen model uses `1x1xK` for decode. The standalone benchmark protocol is therefore not a faithful oracle unless it reproduces the same call context and warmup state as the model.

## Full-model linear attribution

With corrected top-level module hooks:

| Method | No-hook E2E ms | Traced linear sum ms | Backend counts |
| --- | ---: | ---: | --- |
| `sparse_bf16` | `3809.744` in shape trace run | `1449.625` | `{'sparse_bf16': 248}` |
| `manual` | `3753.846` in full trace run | `1140.532` | `{'marlin_nvfp4': 128, 'bf16': 56, 'dense_nvfp4/marlin_nvfp4': 64}` |
| `pred` | `4042.400` in full trace run | `1099.339` | `{'marlin_nvfp4': 56, 'dense_nvfp4/marlin_nvfp4': 128, 'bf16': 64}` |

The hook trace itself slows the model and should only be used for attribution. The stable conclusion is that manual/pred do reduce traced compressible-linear time, but the current manual policy was chosen using bad sparse_bf16 standalone decode numbers.

## Why manual is not an oracle

The manual policy is only an oracle for the benchmark it runs. It is not currently an oracle for full-model E2E because:

1. It benchmarks each phase by repeatedly running isolated synthetic linears.
2. The isolated sparse_bf16 small-m path does not match the real Qwen decode path.
3. The per-shape objective ignores method-dependent full-model residuals outside the measured linear calls.
4. The measured E2E difference between pred and sparse_bf16 on Qwen normal_01 is small enough that run-to-run variation can flip close comparisons.

## Practical Fix

For Qwen3.5-9B policy selection, the manual oracle should benchmark candidates with a model-faithful module replay:

1. Use the exact input ranks seen in the model: prefill `1x16384xK`, decode `1x1xK`.
2. Do not estimate decode by timing an isolated phase after many repeated prefill benchmark iterations.
3. Time a complete mini-scenario per candidate: warmup, one prefill call, then 32 decode calls.
4. Prefer measuring the actual replaced module wrapper used by `replace_linear_with_qwen_predictor_hybrid`, not a separately constructed approximation.

Until this is fixed, `manual` should not be interpreted as a true oracle for Qwen3.5-9B `normal_01`.

## Follow-up: Root Cause of the 3-4 ms sparse_bf16 Decode

The 3-4 ms/call sparse_bf16 decode number comes from the module benchmark protocol, not from the real steady-state model decode.

Raw CUDA-event timing of one sparse_bf16 module shows:

- first small-`m` call: about `35-45 ms`
- subsequent small-`m` calls: about `0.1-0.2 ms`

The old benchmark measured decode with a short average and then used:

```text
prefill + output_tokens * decode_avg
```

Then it multiplied that per-module total by the number of same-shape modules. This repeated a one-time cold-start cost across all tokens and all modules.

A same-shape multi-module test showed the cold-start is mostly global per `(backend, shape, m)`:

```text
decode first per module same shape
module 0: 44.209 ms, then 0.283 ms
module 1: 0.147 ms, then 0.143 ms
module 2: 0.128 ms, then 0.116 ms
module 3: 0.122 ms, then 0.114 ms
```

So the old manual benchmark over-counted sparse_bf16 decode cold-start by a large factor.

`scripts/run_main_hybrid_policy_retest.py` was updated to record:

- `prefill_first_ms`
- `prefill_steady_ms`
- `decode_first_ms`
- `decode_steady_ms`

For ordinary backends, group latency now counts the first cold call once per shape group, then counts the rest as steady calls. For `dense_nvfp4_prefill_marlin_decode`, cold lazy materialization is still counted per module because the shared NVFP4 wrapper materializes backend modules independently per linear.
