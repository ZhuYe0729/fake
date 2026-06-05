# Pure Prefill Hybrid Benchmark — Llama & Qwen3.5

## Scenario

**batch_size=16, input_tokens=1024, output_tokens=1 (pure prefill)**

- Prefill M = batch_size × input_tokens = 16384
- Hybrid = each linear layer independently selects the fastest kernel at M=16384
- Baseline: dense_bf16 = 1.00x, higher = faster
- Data source: module-level kernel benchmarks at M=16384

---

## Combined Result Table

| Method | Llama-2-7B | Llama-3.1-8B | Qwen3.5-9B |
|--------|-----------|-------------|-----------|
| dense_bf16 | 908.33ms (1.0000x) | 984.44ms (1.0000x) | 972.66ms (1.0000x) |
| dense_nvfp4 | 594.99ms (1.5266x) | 644.77ms (1.5268x) | 665.19ms (1.4622x) |
| sparse_bf16 | 468.08ms (1.9406x) | 506.28ms (1.9444x) | 512.77ms (1.8969x) |
| sparse_nvfp4 | 539.89ms (1.6824x) | 584.35ms (1.6847x) | 600.12ms (1.6208x) |
| marlin_nvfp4 | 914.11ms (0.9937x) | 986.10ms (0.9983x) | 989.05ms (0.9834x) |
| hybrid | 413.90ms (2.1945x) | 405.37ms (2.4285x) | 427.24ms (2.2766x) |

---
## Llama-2-7B

| Method | Prefill(ms) | Speedup | Notes |
|--------|------------|---------|-------|
| dense_bf16 | 908.33 | 1.0000x |  |
| dense_nvfp4 | 594.99 | 1.5266x |  |
| sparse_bf16 | 468.08 | 1.9406x |  |
| sparse_nvfp4 | 539.89 | 1.6824x |  |
| marlin_nvfp4 | 914.11 | 0.9937x |  |
| **hybrid** | **413.90** | **2.1945x** | sparse_bf16(160), sparse_nvfp4(64) |

- Hybrid vs dense_bf16: **2.1945x**
- Hybrid vs best single (sparse_bf16 @ 1.9406x): **1.1309x**

### Hybrid Strategy (M=16384)

| Layer | N | K | Best Kernel | Latency(ms) | Count | Subtotal(ms) |
|-------|---|---|------------|------------|-------|-------------|
| mlp.down_proj | 4096 | 11008 | sparse_bf16 | 3.2048 | 32 | 102.553 |
| mlp.gate_proj | 11008 | 4096 | sparse_nvfp4 | 2.3554 | 32 | 75.372 |
| mlp.up_proj | 11008 | 4096 | sparse_nvfp4 | 2.3447 | 32 | 75.029 |
| self_attn.k_proj | 4096 | 4096 | sparse_bf16 | 1.2574 | 32 | 40.238 |
| self_attn.o_proj | 4096 | 4096 | sparse_bf16 | 1.2574 | 32 | 40.236 |
| self_attn.q_proj | 4096 | 4096 | sparse_bf16 | 1.2576 | 32 | 40.244 |
| self_attn.v_proj | 4096 | 4096 | sparse_bf16 | 1.2573 | 32 | 40.233 |

---
## Llama-3.1-8B

| Method | Prefill(ms) | Speedup | Notes |
|--------|------------|---------|-------|
| dense_bf16 | 984.44 | 1.0000x |  |
| dense_nvfp4 | 644.77 | 1.5268x |  |
| sparse_bf16 | 506.28 | 1.9444x |  |
| sparse_nvfp4 | 584.35 | 1.6847x |  |
| marlin_nvfp4 | 986.10 | 0.9983x |  |
| **hybrid** | **405.37** | **2.4285x** | sparse_bf16(160), sparse_nvfp4(64) |

- Hybrid vs dense_bf16: **2.4285x**
- Hybrid vs best single (sparse_bf16 @ 1.9444x): **1.2489x**

### Hybrid Strategy (M=16384)

| Layer | N | K | Best Kernel | Latency(ms) | Count | Subtotal(ms) |
|-------|---|---|------------|------------|-------|-------------|
| mlp.down_proj | 4096 | 14336 | sparse_bf16 | 4.1688 | 32 | 133.400 |
| mlp.gate_proj | 14336 | 4096 | sparse_nvfp4 | 2.6037 | 32 | 83.317 |
| mlp.up_proj | 14336 | 4096 | sparse_nvfp4 | 2.5985 | 32 | 83.153 |
| self_attn.k_proj | 1024 | 4096 | sparse_bf16 | 0.3866 | 32 | 12.370 |
| self_attn.o_proj | 4096 | 4096 | sparse_bf16 | 1.2618 | 32 | 40.376 |
| self_attn.q_proj | 4096 | 4096 | sparse_bf16 | 1.2577 | 32 | 40.248 |
| self_attn.v_proj | 1024 | 4096 | sparse_bf16 | 0.3909 | 32 | 12.508 |

---
## Qwen3.5-9B

| Method | Prefill(ms) | Speedup | Notes |
|--------|------------|---------|-------|
| dense_bf16 | 972.66 | 1.0000x |  |
| dense_nvfp4 | 665.19 | 1.4622x |  |
| sparse_bf16 | 512.77 | 1.8969x | 16 layers fallback to dense_bf16 |
| sparse_nvfp4 | 600.12 | 1.6208x |  |
| marlin_nvfp4 | 989.05 | 0.9834x | 48 layers fallback to dense_bf16 |
| **hybrid** | **427.24** | **2.2766x** | sparse_bf16(136), sparse_nvfp4(96), marlin_nvfp4(16) |

- Hybrid vs dense_bf16: **2.2766x**
- Hybrid vs best single (sparse_bf16 @ 1.8969x): **1.2002x**

### Hybrid Strategy (M=16384)

| Layer | N | K | Best Kernel | Latency(ms) | Count | Subtotal(ms) |
|-------|---|---|------------|------------|-------|-------------|
| linear_attn.in_proj_a | 32 | 4096 | sparse_bf16 | 0.1215 | 24 | 2.916 |
| linear_attn.in_proj_b | 32 | 4096 | sparse_bf16 | 0.1203 | 24 | 2.888 |
| linear_attn.in_proj_qkv | 8192 | 4096 | sparse_nvfp4 | 2.1395 | 24 | 51.347 |
| linear_attn.in_proj_z | 4096 | 4096 | sparse_bf16 | 1.2686 | 24 | 30.447 |
| linear_attn.out_proj | 4096 | 4096 | sparse_bf16 | 1.2669 | 24 | 30.406 |
| mlp.down_proj | 4096 | 12288 | sparse_bf16 | 3.5990 | 32 | 115.169 |
| mlp.gate_proj | 12288 | 4096 | sparse_nvfp4 | 2.4551 | 32 | 78.563 |
| mlp.up_proj | 12288 | 4096 | sparse_nvfp4 | 2.4479 | 32 | 78.331 |
| self_attn.k_proj | 1024 | 4096 | marlin_nvfp4 | 0.6204 | 8 | 4.963 |
| self_attn.o_proj | 4096 | 4096 | sparse_bf16 | 1.2733 | 8 | 10.186 |
| self_attn.q_proj | 8192 | 4096 | sparse_nvfp4 | 2.1332 | 8 | 17.065 |
| self_attn.v_proj | 1024 | 4096 | marlin_nvfp4 | 0.6197 | 8 | 4.958 |

*Generated: 2026-06-03 | GPU: NVIDIA RTX 5090 32GB | PyTorch: 2.9.0+cu128 | CUDA: 12.8*
