# Qwen3.5-2B Kernel-Driven Hybrid Revision

Date: 2026-06-02

## Why the Previous Manual Hybrid Was Slower

The previous `manual_hybrid_m1/m4/m8/m16` policy was not solved by summing the measured/predicted kernel latency for every Qwen3.5-2B linear shape. It used a coarse rule:

- prefill: many layers use `sparse_nvfp4` or `dense_nvfp4`
- decode: most layers use `marlin_nvfp4`
- small projections such as `linear_attn.in_proj_a/b` and some K/V projections fall back to BF16

The actual end-to-end stage results show this is not the best policy for Qwen3.5-2B:

| Scene | Dense Prefill ms | Sparse BF16 Prefill ms | Dense Decode/Token ms | Marlin Decode/Token ms |
|---|---:|---:|---:|---:|
| A_long_context | 546.61 | 522.70 | 20.85 | 23.96 |
| B_batched_rag | 645.06 | 546.14 | 21.18 | 24.69 |
| C_medium_batch | 613.75 | 516.55 | 21.13 | 24.60 |
| D_high_batch_short | 599.32 | 501.94 | 21.32 | 24.49 |
| E_long_generation | 193.71 | 193.59 | 20.86 | 23.96 |

So the previous policy became slower because it switched the wrong stage:

- `marlin_nvfp4` is slower than `dense` in the measured Qwen3.5-2B decode path.
- `sparse_bf16` is faster than `dense` in the measured Qwen3.5-2B prefill path.
- `sparse_nvfp4` does not provide the expected end-to-end prefill benefit for this 2B model, even though the standalone kernel table suggests it can win for some large `(M,N,K)` shapes.

## Qwen3.5-2B Linear Shapes

The language model has 186 replaceable linear layers:

| Linear group | Count | N | K |
|---|---:|---:|---:|
| `linear_attn.in_proj_a` | 18 | 16 | 2048 |
| `linear_attn.in_proj_b` | 18 | 16 | 2048 |
| `linear_attn.in_proj_qkv` | 18 | 6144 | 2048 |
| `linear_attn.in_proj_z` | 18 | 2048 | 2048 |
| `linear_attn.out_proj` | 18 | 2048 | 2048 |
| `mlp.gate_proj` | 24 | 6144 | 2048 |
| `mlp.up_proj` | 24 | 6144 | 2048 |
| `mlp.down_proj` | 24 | 2048 | 6144 |
| `self_attn.q_proj` | 6 | 4096 | 2048 |
| `self_attn.k_proj` | 6 | 512 | 2048 |
| `self_attn.v_proj` | 6 | 512 | 2048 |
| `self_attn.o_proj` | 6 | 2048 | 2048 |

This is why the global benchmark report cannot be applied directly. Qwen3.5-2B has many small or asymmetric shapes, especially `(16,2048)` and `(512,2048)`.

## Revised Safe Hybrid Principle

For a paper result that should not regress, use a stage-constrained policy:

- Prefill: use the measured prefill winner, `sparse_bf16`, but keep tiny `N=16` projections in `dense`.
- Decode: keep `dense`, because it is the measured decode winner in all 5 scenes.

This still demonstrates two hybrid dimensions:

- Workload hybrid: prefill and decode use different methods.
- Shape hybrid: tiny `linear_attn.in_proj_a/b` stay dense while larger linear layers use sparse BF16 during prefill.

## Strongest Scene to Test

The strongest controlled scene should make prefill dominate and keep decode short:

| Proposed Scene | Batch | Input Tokens | Output Tokens | Reason |
|---|---:|---:|---:|---|
| `F_prefill_dominant_short_decode` | 16 | 1024 | 1 | Uses the measured `M_prefill=16384` regime where `sparse_bf16` prefill is much faster than dense, while only paying one dense decode step. |

Using the already measured D scene stage numbers:

- dense estimate for `B=16,I=1024,O=1`: `599.32 + 21.61 = 620.93 ms`
- revised hybrid estimate: `501.94 + 21.61 = 523.55 ms`
- expected improvement: about `15.7%`

This is much more favorable than `O=128`, where decode dominates:

- dense D measured total: `3307.15 ms`
- sparse-prefill + dense-decode estimate: `501.94 + 2707.83 = 3209.78 ms`
- expected improvement: about `2.9%`

## Revised Hybrid Scheme

For `F_prefill_dominant_short_decode`:

| Linear group | Prefill backend | Decode backend |
|---|---|---|
| `linear_attn.in_proj_a` | `bf16` | `bf16` |
| `linear_attn.in_proj_b` | `bf16` | `bf16` |
| `linear_attn.in_proj_qkv` | `sparse_bf16` | `bf16` |
| `linear_attn.in_proj_z` | `sparse_bf16` | `bf16` |
| `linear_attn.out_proj` | `sparse_bf16` | `bf16` |
| `mlp.gate_proj` | `sparse_bf16` | `bf16` |
| `mlp.up_proj` | `sparse_bf16` | `bf16` |
| `mlp.down_proj` | `sparse_bf16` | `bf16` |
| `self_attn.q_proj` | `sparse_bf16` | `bf16` |
| `self_attn.k_proj` | `sparse_bf16` | `bf16` |
| `self_attn.v_proj` | `sparse_bf16` | `bf16` |
| `self_attn.o_proj` | `sparse_bf16` | `bf16` |

This is the conservative speed-only hybrid for Qwen3.5-2B. It is not the same-weight NVFP4 hybrid; it is the strongest currently supported stage/shape hybrid suggested by the completed main results.

## What to Test Next

Add a new manual method, for example `manual_hybrid_prefill_sparse_decode_dense`, with:

- `decode_m_threshold = 16`
- decode backend always `bf16`
- prefill backend `bf16` only for `linear_attn.in_proj_a/b`
- prefill backend `sparse_bf16` for all other replaceable linear layers

Then test:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/prepare_qwen3_5_kernel_checkpoint.py \
  --variant 2B \
  --method manual_hybrid_prefill_sparse_decode_dense \
  --dtype bf16 \
  --output artifacts/results/main/qwen3_5_2b/checkpoints/manual_hybrid_prefill_sparse_decode_dense/model.pt

CUDA_VISIBLE_DEVICES=0 python scripts/bench_qwen3_5_speed.py \
  --variant 2B \
  --method manual_hybrid_prefill_sparse_decode_dense \
  --checkpoint artifacts/results/main/qwen3_5_2b/checkpoints/manual_hybrid_prefill_sparse_decode_dense/model.pt \
  --batch-sizes 16 \
  --input-tokens 1024 \
  --output-tokens 1 \
  --warmup 5 \
  --iters 20 \
  --output-csv artifacts/results/main/qwen3_5_2b/scenes/F_prefill_dominant_short_decode/manual_hybrid_prefill_sparse_decode_dense/speed.csv
```

Also run the dense baseline for the same scene:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/bench_qwen3_5_speed.py \
  --variant 2B \
  --method dense \
  --batch-sizes 16 \
  --input-tokens 1024 \
  --output-tokens 1 \
  --warmup 5 \
  --iters 20 \
  --output-csv artifacts/results/main/qwen3_5_2b/scenes/F_prefill_dominant_short_decode/dense/speed.csv
```
