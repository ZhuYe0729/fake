# Qwen3.5 4B/9B/27B Module Kernel Benchmarks

This benchmark measures packaged `Linear.forward` latency for Qwen3.5 compressible
linear shapes while sweeping token dimension `M`.

CSV semantics:

- `M = tokens`
- `N = out_features`
- `K = in_features`
- measured callable is `module(x)` with input shape `(1, M, K)`
- dense NVFP4 activation packing is included in `forward`
- sparse BF16/NVFP4 use padded wrappers matching Qwen runtime

Default `M` sweep:

```text
1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384
```

## Parallel Command

Run 4B/9B/27B on physical GPUs 1/2/3 and then generate curve plots:

```bash
conda activate cospaq

GPU_LIST="1 2 3" bash scripts/run_qwen3_5_module_kernel_benchmarks_4b_9b_27b.sh
```

Outputs:

```text
artifacts/results/benchmarks/module/Qwen3.5-4B/kernel/qwen35_4b_module_kernel_curves.csv
artifacts/results/benchmarks/module/Qwen3.5-4B/kernel/qwen35_4b_module_kernel_latency_curves.png
artifacts/results/benchmarks/module/Qwen3.5-4B/kernel/qwen35_4b_module_kernel_latency_curves.pdf

artifacts/results/benchmarks/module/Qwen3.5-9B/kernel/qwen35_9b_module_kernel_curves.csv
artifacts/results/benchmarks/module/Qwen3.5-9B/kernel/qwen35_9b_module_kernel_latency_curves.png
artifacts/results/benchmarks/module/Qwen3.5-9B/kernel/qwen35_9b_module_kernel_latency_curves.pdf

artifacts/results/benchmarks/module/Qwen3.5-27B/kernel/qwen35_27b_module_kernel_curves.csv
artifacts/results/benchmarks/module/Qwen3.5-27B/kernel/qwen35_27b_module_kernel_latency_curves.png
artifacts/results/benchmarks/module/Qwen3.5-27B/kernel/qwen35_27b_module_kernel_latency_curves.pdf
```

## Single Model Commands

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/bench_qwen3_5_module_kernels.py \
  --model-name Qwen3.5-4B \
  --gpu 0 \
  --output artifacts/results/benchmarks/module/Qwen3.5-4B/kernel/qwen35_4b_module_kernel_curves.csv

python scripts/visualize_qwen3_5_module_kernel_curves.py \
  --input artifacts/results/benchmarks/module/Qwen3.5-4B/kernel/qwen35_4b_module_kernel_curves.csv \
  --output artifacts/results/benchmarks/module/Qwen3.5-4B/kernel/qwen35_4b_module_kernel_latency_curves.png
```

Replace `Qwen3.5-4B/qwen35_4b` with `Qwen3.5-9B/qwen35_9b` or
`Qwen3.5-27B/qwen35_27b` for the other models.

## Shape Summary

### Qwen3.5-4B

| Linear group | Count | N | K |
|---|---:|---:|---:|
| `linear_attn.in_proj_a` | 24 | 32 | 2560 |
| `linear_attn.in_proj_b` | 24 | 32 | 2560 |
| `linear_attn.in_proj_qkv` | 24 | 8192 | 2560 |
| `linear_attn.in_proj_z` | 24 | 4096 | 2560 |
| `linear_attn.out_proj` | 24 | 2560 | 4096 |
| `mlp.gate_proj` | 32 | 9216 | 2560 |
| `mlp.up_proj` | 32 | 9216 | 2560 |
| `mlp.down_proj` | 32 | 2560 | 9216 |
| `self_attn.q_proj` | 8 | 8192 | 2560 |
| `self_attn.k_proj` | 8 | 1024 | 2560 |
| `self_attn.v_proj` | 8 | 1024 | 2560 |
| `self_attn.o_proj` | 8 | 2560 | 4096 |

### Qwen3.5-9B

| Linear group | Count | N | K |
|---|---:|---:|---:|
| `linear_attn.in_proj_a` | 24 | 32 | 4096 |
| `linear_attn.in_proj_b` | 24 | 32 | 4096 |
| `linear_attn.in_proj_qkv` | 24 | 8192 | 4096 |
| `linear_attn.in_proj_z` | 24 | 4096 | 4096 |
| `linear_attn.out_proj` | 24 | 4096 | 4096 |
| `mlp.gate_proj` | 32 | 12288 | 4096 |
| `mlp.up_proj` | 32 | 12288 | 4096 |
| `mlp.down_proj` | 32 | 4096 | 12288 |
| `self_attn.q_proj` | 8 | 8192 | 4096 |
| `self_attn.k_proj` | 8 | 1024 | 4096 |
| `self_attn.v_proj` | 8 | 1024 | 4096 |
| `self_attn.o_proj` | 8 | 4096 | 4096 |

### Qwen3.5-27B

| Linear group | Count | N | K |
|---|---:|---:|---:|
| `linear_attn.in_proj_a` | 48 | 48 | 5120 |
| `linear_attn.in_proj_b` | 48 | 48 | 5120 |
| `linear_attn.in_proj_qkv` | 48 | 10240 | 5120 |
| `linear_attn.in_proj_z` | 48 | 6144 | 5120 |
| `linear_attn.out_proj` | 48 | 5120 | 6144 |
| `mlp.gate_proj` | 64 | 17408 | 5120 |
| `mlp.up_proj` | 64 | 17408 | 5120 |
| `mlp.down_proj` | 64 | 5120 | 17408 |
| `self_attn.q_proj` | 16 | 12288 | 5120 |
| `self_attn.k_proj` | 16 | 1024 | 5120 |
| `self_attn.v_proj` | 16 | 1024 | 5120 |
| `self_attn.o_proj` | 16 | 5120 | 6144 |
