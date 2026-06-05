# Qwen3.5-2B Module Kernel Benchmark

This benchmark measures packaged `Linear.forward` latency for the unique compressible
linear shapes in Qwen3.5-2B while sweeping token dimension `M`.

It is intentionally different from raw GEMM kernel benchmarks:

- measured callable: `module(x)`
- input shape: `(1, M, K)`
- CSV shape semantics: `M = tokens`, `N = out_features`, `K = in_features`
- dense NVFP4 includes activation packing in `forward`
- sparse BF16/NVFP4 use the padded wrappers used by Qwen runtime, so small `M`
  values are padded before calling the underlying sparse kernels

## Command

```bash
conda activate cospaq

CUDA_VISIBLE_DEVICES=0 python scripts/bench_qwen3_5_2b_module_kernels.py \
  --gpu 0 \
  --output artifacts/results/benchmarks/module/Qwen3.5-2B/kernel/qwen35_2b_module_kernel_curves.csv \
  --warmup 5 \
  --iters 20
```

If binding to another physical GPU, keep `--gpu 0` when `CUDA_VISIBLE_DEVICES`
contains a single device:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/bench_qwen3_5_2b_module_kernels.py --gpu 0
```

## Tested Linear Shapes

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

Default `M` sweep:

```text
1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384
```

Default kernels:

```text
dense_bf16 dense_nvfp4 sparse_bf16 sparse_nvfp4 marlin_nvfp4
```
