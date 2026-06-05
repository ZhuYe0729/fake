# Qwen3.5-2B Main Result Analysis

Date: 2026-06-02

## Scope

Analyzed files:

- `artifacts/results/main/qwen3_5_2b/scenes/*/*/speed.csv`

Each method has one measured row per scene, so the comparison below is a single-run result without variance.

## End-to-End Ranking

Total latency is computed as `prefill_ms + decode_total_ms`.

| Scene | Shape | Best Method | Best Total ms | Hybrid Method | Hybrid Total ms | Hybrid vs Best Single |
|---|---:|---|---:|---|---:|---:|
| A_long_context | B=1, I=8192, O=512 | dense | 11201.24 | manual_hybrid_m1 | 13891.34 | +24.0% |
| B_batched_rag | B=4, I=4096, O=512 | dense | 11466.23 | manual_hybrid_m4 | 13380.64 | +16.7% |
| C_medium_batch | B=8, I=2048, O=256 | dense | 6001.49 | manual_hybrid_m8 | 6675.99 | +11.2% |
| D_high_batch_short | B=16, I=1024, O=128 | dense | 3307.15 | manual_hybrid_m16 | 3841.59 | +16.2% |
| E_long_generation | B=1, I=2048, O=1024 | dense | 21538.83 | manual_hybrid_m1 | 25617.76 | +18.9% |

Current result: the implemented manual hybrid methods are slower than the best single method in all 5 scenes.

## Per-Stage Winners

| Scene | Best Prefill | Prefill ms | Best Decode/Token | Decode/Token ms | Best Total |
|---|---|---:|---|---:|---|
| A_long_context | sparse_bf16 | 522.70 | dense | 20.85 | dense |
| B_batched_rag | sparse_bf16 | 546.14 | dense | 21.18 | dense |
| C_medium_batch | sparse_bf16 | 516.55 | dense | 21.13 | dense |
| D_high_batch_short | sparse_bf16 | 501.94 | dense | 21.32 | dense |
| E_long_generation | sparse_bf16 | 193.59 | dense | 20.86 | dense |

This does show workload-level heterogeneity:

- Prefill prefers `sparse_bf16` in every scene.
- Decode prefers `dense` in every scene.

However, the current hybrid policies mostly use `marlin_nvfp4` for decode, and `marlin_nvfp4` is slower than `dense` for these end-to-end decode measurements.

## Full Scene Tables

### A_long_context, B=1, I=8192, O=512

| Method | Total ms | Prefill ms | Decode Total ms | Decode/Token ms | Tokens/s |
|---|---:|---:|---:|---:|---:|
| dense | 11201.24 | 546.61 | 10654.63 | 20.85 | 47.96 |
| marlin_nvfp4 | 12784.29 | 538.61 | 12245.68 | 23.96 | 41.73 |
| manual_hybrid_m1 | 13891.34 | 637.33 | 13254.02 | 25.94 | 38.55 |
| sparse_bf16 | 14880.93 | 522.70 | 14358.23 | 28.10 | 35.59 |
| dense_nvfp4 | 19637.55 | 620.07 | 19017.48 | 37.22 | 26.87 |
| sparse_nvfp4 | 24555.31 | 634.20 | 23921.11 | 46.81 | 21.36 |

### B_batched_rag, B=4, I=4096, O=512

| Method | Total ms | Prefill ms | Decode Total ms | Decode/Token ms | Tokens/s |
|---|---:|---:|---:|---:|---:|
| dense | 11466.23 | 645.06 | 10821.18 | 21.18 | 188.89 |
| marlin_nvfp4 | 13266.99 | 650.50 | 12616.49 | 24.69 | 162.01 |
| manual_hybrid_m4 | 13380.64 | 628.30 | 12752.34 | 24.96 | 160.28 |
| sparse_bf16 | 15094.34 | 546.14 | 14548.20 | 28.47 | 140.50 |
| dense_nvfp4 | 20062.23 | 631.87 | 19430.35 | 38.02 | 105.20 |
| sparse_nvfp4 | 24436.62 | 641.11 | 23795.52 | 46.57 | 85.90 |

### C_medium_batch, B=8, I=2048, O=256

| Method | Total ms | Prefill ms | Decode Total ms | Decode/Token ms | Tokens/s |
|---|---:|---:|---:|---:|---:|
| dense | 6001.49 | 613.75 | 5387.75 | 21.13 | 378.64 |
| manual_hybrid_m8 | 6675.99 | 596.00 | 6079.98 | 23.84 | 335.53 |
| sparse_bf16 | 6755.65 | 516.55 | 6239.10 | 24.47 | 326.97 |
| marlin_nvfp4 | 6892.03 | 620.16 | 6271.88 | 24.60 | 325.26 |
| dense_nvfp4 | 10246.89 | 600.75 | 9646.15 | 37.83 | 211.48 |
| sparse_nvfp4 | 12463.99 | 608.41 | 11855.58 | 46.49 | 172.07 |

### D_high_batch_short, B=16, I=1024, O=128

| Method | Total ms | Prefill ms | Decode Total ms | Decode/Token ms | Tokens/s |
|---|---:|---:|---:|---:|---:|
| dense | 3307.15 | 599.32 | 2707.83 | 21.32 | 750.42 |
| sparse_bf16 | 3629.14 | 501.94 | 3127.19 | 24.62 | 649.78 |
| marlin_nvfp4 | 3715.91 | 606.37 | 3109.54 | 24.48 | 653.47 |
| manual_hybrid_m16 | 3841.59 | 581.14 | 3260.44 | 25.67 | 623.23 |
| dense_nvfp4 | 5454.96 | 586.85 | 4868.11 | 38.33 | 417.41 |
| sparse_nvfp4 | 6509.34 | 594.26 | 5915.09 | 46.57 | 343.53 |

### E_long_generation, B=1, I=2048, O=1024

| Method | Total ms | Prefill ms | Decode Total ms | Decode/Token ms | Tokens/s |
|---|---:|---:|---:|---:|---:|
| dense | 21538.83 | 193.71 | 21345.11 | 20.86 | 47.93 |
| marlin_nvfp4 | 24713.03 | 197.06 | 24515.97 | 23.96 | 41.73 |
| manual_hybrid_m1 | 25617.76 | 212.33 | 25405.43 | 24.83 | 40.27 |
| sparse_bf16 | 28402.48 | 193.59 | 28208.89 | 27.58 | 36.27 |
| dense_nvfp4 | 38954.09 | 214.74 | 38739.35 | 37.87 | 26.41 |
| sparse_nvfp4 | 47563.42 | 219.63 | 47343.79 | 46.28 | 21.61 |

## Interpretation

The current main experiment is useful, but it does not yet support the intended headline that the implemented manual hybrid is faster than all single-method baselines.

What it does support:

- Different stages prefer different methods: prefill is best with `sparse_bf16`, decode is best with `dense`.
- Weight-only `marlin_nvfp4` is consistently better than dense/sparse NVFP4 W4A4 kernels for decode, but still slower than BF16 dense in these end-to-end runs.
- Dense/sparse NVFP4 W4A4 kernels are currently too slow for this Qwen3.5-2B end-to-end benchmark.

Likely reason:

- The manual hybrid policies were designed from per-linear kernel intuition, but the measured end-to-end decode path is dominated by cases where `dense` is faster than `marlin_nvfp4`.
- The current hybrid includes mixed backend dispatch overhead and uses compressed kernels for layers where the end-to-end path does not benefit.

## Recommended Next Step

For a speed-only main experiment, the next manual hybrid should be tested as:

- Prefill: use `sparse_bf16` for the linear groups where it gives the observed prefill advantage.
- Decode: use `dense` for most or all linear layers, because it is the measured decode winner.

This would test the actual heterogeneity shown by the current results: `sparse_bf16` for large-M prefill and `dense` for small-M decode. If the purpose is specifically to demonstrate NVFP4 same-weight switching, these results suggest that the current Qwen3.5-2B setting is not favorable unless the NVFP4 kernels are improved or a different model/shape regime is selected from layer-level benchmarks.

## G_b1_i8192_o32 Result

Scene:

- `batch=1`
- `input_tokens=8192`
- `output_tokens=32`
- prefill GEMM `M=8192`
- decode GEMM `M=1`

This scene was used to test the recommended dense NVFP4 same-weight hybrid `hybrid_nvfp4_major`, where large linear layers use W4A4 `dense_nvfp4` for prefill and W4A16 `marlin_nvfp4` for decode, while small layers stay BF16.

| Method | Total ms | Prefill ms | Decode Total ms | Decode/Token ms | Tokens/s |
|---|---:|---:|---:|---:|---:|
| dense | 1266.86 | 578.37 | 688.49 | 22.21 | 45.03 |
| marlin_nvfp4 | 1344.67 | 562.99 | 781.68 | 25.22 | 39.66 |
| hybrid_nvfp4_major | 1386.65 | 617.30 | 769.35 | 24.82 | 40.29 |
| sparse_bf16 | 1509.56 | 567.38 | 942.19 | 30.39 | 32.90 |
| dense_nvfp4 | 1920.29 | 664.17 | 1256.13 | 40.52 | 24.68 |
| sparse_nvfp4 | 2060.01 | 629.44 | 1430.57 | 46.15 | 21.67 |

Key comparisons:

- `hybrid_nvfp4_major` is slower than dense by `9.5%` in total latency.
- `hybrid_nvfp4_major` is slower than full `marlin_nvfp4` by `3.1%` in total latency.
- `hybrid_nvfp4_major` is faster than `dense_nvfp4`, `sparse_bf16`, and `sparse_nvfp4`.
- Dense still has the best decode/token latency: `22.21 ms/token`.
- Marlin has the best prefill latency in this scene: `562.99 ms`, but dense is very close at `578.37 ms`.

Interpretation:

The result does not validate the kernel-level prediction that W4A4/W4A16 dense NVFP4 switching should beat dense end-to-end for Qwen3.5-2B. The main mismatch is that end-to-end decode still favors BF16 dense, while standalone GEMM predicted Marlin advantages for many large linear shapes. Also, `hybrid_nvfp4_major` prefill is slower than both dense and full Marlin, indicating that W4A4 dense NVFP4 prefill is not paying off in the actual model path.

For the paper, this scene should not be used as the main positive speed result for Qwen3.5-2B. It is useful as evidence that kernel-level routing must be calibrated by end-to-end stage measurements.

## Why Kernel-Level Predictions Did Not Match End-to-End

The mismatch is not because `batch=1,input=8192,output=32` was measured with the wrong prefill/decode `M`.

The benchmark script does:

- Prefill: `model(input_ids=[1,8192], use_cache=True)`, so linear GEMM `M=8192` for normal token-wise linear layers.
- Decode: `model(input_ids=[1,1], past_key_values=..., use_cache=True)`, so linear GEMM `M=1` for normal token-wise linear layers.

The replacement policy is also installed as intended:

- `hybrid_nvfp4_major` replaces 138 large linear layers.
- It keeps 48 layers as BF16: `linear_attn.in_proj_a/b` and `self_attn.k/v_proj`.

The real issue is that `5_kernel_comprehensive` is a GEMM-only benchmark, not a packaged `Linear.forward` benchmark under the full model path.

### 1. Dense NVFP4 GEMM timing excludes activation packing

In `bench_5_kernels_comprehensive.py`, dense NVFP4 preparation is done before timing:

- `pack_nvfp4_a(x_bf16)`
- `pack_nvfp4_b(weight_bf16)`
- scale/alpha preparation

The timed callable then only runs:

- `nvfp4_gemm_bf16(a_packed, a_sf, b_packed, b_sf, alpha, m, n, k)`

But actual `NVFP4Linear.forward` does activation packing every time:

- flatten/contiguous input
- `pack_nvfp4_a(x_flat)`
- alpha computation
- `nvfp4_gemm_bf16(...)`
- output reshape

Therefore the kernel table overestimates W4A4 dense NVFP4 speed in model inference. This directly explains why `hybrid_nvfp4_major` prefill is slower than expected: prefill uses W4A4 dense NVFP4 on 138 large linear layers, and each one pays activation packing cost in the actual forward.

### 2. Sparse NVFP4 GEMM timing also excludes activation packing

The same issue applies to sparse NVFP4:

- benchmark precomputes `pack_sparse_nvfp4_b(x_bf16)` outside timing
- actual `SparseNVFP4Linear.forward` packs activation inside every forward

So sparse NVFP4 kernel-level wins can disappear in end-to-end model inference.

### 3. Marlin W4A16 is closer to the model path, but decode has low linear share

Marlin W4A16 does not quantize activations, so its kernel benchmark is closer to actual module forward. However, Qwen3.5-2B decode is not dominated by the replaceable linear layers.

For `B=1,I=8192,O=32`:

- Dense decode end-to-end: `22.21 ms/token`
- Kernel-level estimate for replaceable dense linear work: about `3.26 ms/token`
- Kernel-level estimate for replaceable Marlin linear work: about `2.15 ms/token`

Even under the optimistic kernel model, Marlin only saves about `1.1 ms/token` inside a `22 ms/token` decode step. That saving is small enough to be erased by wrapper dispatch, tensor reshape/contiguous overhead, cache update, attention/linear-attn core, norm, activation, and `lm_head`.

### 4. Hybrid lazy conversion may affect early measured steps

`hybrid_nvfp4_major` stores canonical dense NVFP4 weights and lazily constructs CUTLASS/Marlin module objects on first use. The benchmark warmup should mostly absorb this, but it is still another difference from the raw GEMM table. It should be verified with a breakdown or repeated runs if this method remains under consideration.

### What this means

The kernel benchmark is useful, but it currently answers:

> Given already prepared/packed operands, which GEMM kernel is fastest?

The model benchmark answers:

> Given real hidden states inside Qwen3.5-2B, including packing, reshape, dispatch, cache, attention, and lm_head overhead, which method is fastest end-to-end?

These are not identical measurements. For routing decisions, the missing measurement is a module-level benchmark:

- instantiate real `NVFP4Linear`, `MarlinNVFP4Linear`, `SparseBF16Linear`, `SparseNVFP4Linear`
- call `module(x)` with input shapes matching Qwen3.5-2B
- include activation packing and wrapper overhead
- report both `module.forward` latency and GEMM-only latency

Until that is added, the safest routing policy should be based on end-to-end stage measurements or module-level forward measurements, not GEMM-only speed.
