# Qwen3.5 Shape-Workload Hybrid 端到端推理对比

## 场景

**batch_size=32, input_tokens=512, output_tokens=32**

- Prefill: M = batch_size × input_tokens ≈ 16384 (large batch)
- Decode: M = batch_size = 32 (small batch, one token per request per step)
- 基准: dense_bf16 = 1.00x，越高越快

---

## 全系列 E2E 真实推理对比

所有数据来自真实压缩模型端到端推理（load dense → in-memory kernel replace → prefill + KV cache decode）。

### Qwen3.5-9B ✅ (hybrid 最优)

GPU: 1× RTX 5090 (32GB)

| Method | Prefill(ms) | Decode×32(ms) | E2E(ms) | Speedup |
|--------|------------|---------------|---------|---------|
| dense (bf16) | 3960 | 1975 | 5934 | **1.00x** |
| dense_nvfp4 | 3512 | 18759 | 22272 | 0.27x |
| sparse_bf16 | 3077 | 2482 | 5558 | 1.07x |
| sparse_nvfp4 | 3343 | 18740 | 22083 | 0.27x |
| marlin_nvfp4 | 4015 | 1519 | 5534 | 1.07x |
| **shape_workload_hybrid** | **3034** | **1526** | **4560** | **1.30x** |

- Hybrid 比最优单一方法 (marlin_nvfp4 @1.07x) 快 **1.21x**
- Hybrid 比 dense_bf16 基线快 **1.30x**
- dense_nvfp4 / sparse_nvfp4 在 decode (M=32) 时全面崩溃：小 batch 下 NVFP4 activation packing overhead 远超计算收益

### Qwen3.5-4B

GPU: 1× RTX 5090 (32GB)

| Method | Prefill(ms) | Decode×32(ms) | E2E(ms) | Speedup |
|--------|------------|---------------|---------|---------|
| dense (bf16) | 1331 | 1371 | 2702 | **1.00x** |
| dense_nvfp4 | 1267 | 2613 | 3880 | 0.70x |
| sparse_bf16 | 1359 | 1851 | 3209 | 0.84x |
| sparse_nvfp4 | 1227 | 2755 | 3982 | 0.68x |
| marlin_nvfp4 | 1349 | 1461 | 2810 | 0.96x |
| shape_workload_hybrid | 1226 | 1515 | 2740 | 0.99x |

Hybrid 接近但未超越 dense（策略为9B优化，4B维度不同）。

### Qwen3.5-2B

GPU: 1× RTX 5090 (32GB)

| Method | Prefill(ms) | Decode×32(ms) | E2E(ms) | Speedup |
|--------|------------|---------------|---------|---------|
| dense (bf16) | 534 | 1108 | 1642 | **1.00x** |
| marlin_nvfp4 | 539 | 1147 | 1686 | 0.97x |
| shape_workload_hybrid | 569 | 1180 | 1750 | 0.94x |

Hybrid 慢于 dense（N=16 tiny层上 sparse/quant overhead > 收益）。

### Qwen3.5-0.8B

GPU: 1× RTX 5090 (32GB)

| Method | Prefill(ms) | Decode×32(ms) | E2E(ms) | Speedup |
|--------|------------|---------------|---------|---------|
| dense (bf16) | 371 | 1014 | 1385 | **1.00x** |
| marlin_nvfp4 | 371 | 1050 | 1421 | 0.97x |
| shape_workload_hybrid | 492 | 1127 | 1619 | 0.86x |

Hybrid 显著慢于 dense（维度太小，kernel overhead 无法摊销）。

### Qwen3.5-27B ⚠️ (多卡，部分方法不可用)

GPU: 4× RTX 5090 (128GB), batch=1 (显存限制)

| Method | Prefill(ms) | Decode×32(ms) | E2E(ms) | Speedup |
|--------|------------|---------------|---------|---------|
| dense (bf16) | 625 | 3504 | 4129 | **1.00x** |
| dense_nvfp4 | 482 | 4774 | 5256 | 0.79x |
| sparse_bf16 | 1998 | 6080 | 8078 | 0.51x |
| sparse_nvfp4 | 499 | 5856 | 6355 | 0.65x |
| marlin_nvfp4 | — | — | — | kernel crash |
| shape_workload_hybrid | — | — | — | dual-backend OOM |

Blockers:
- marlin_nvfp4: 转换正常(400/496层)，多卡forward时 illegal memory access（单层12/12全通过，kernel代码级bug）
- hybrid: QwenManualHybridLinear 需双backend，显存翻倍，4卡OOM
- sparse_bf16: 8卡cuSPARSELt报错，4卡正常但prefill极慢(1998ms vs dense 625ms)

---

## Shape-Workload Hybrid 策略详情

基于 `artifacts/results/benchmarks/module/Qwen3.5-9B/kernel/` 的实测数据，
对每个 (linear_group, M) 选择延迟最低的 kernel。

### Prefill 阶段 (M=16384) — 3种 kernel

| 层类型 | 最优 Kernel | 单层延迟 | 层数 | 小计 |
|--------|------------|---------|------|------|
| `in_proj_a` | sparse_bf16 | 0.12ms | 24 | 2.9ms |
| `in_proj_b` | sparse_bf16 | 0.12ms | 24 | 2.9ms |
| `in_proj_qkv` | **sparse_nvfp4** | 2.14ms | 24 | 51.3ms |
| `in_proj_z` | sparse_bf16 | 1.27ms | 24 | 30.4ms |
| `out_proj` | sparse_bf16 | 1.27ms | 24 | 30.4ms |
| `mlp.down_proj` | sparse_bf16 | 3.60ms | 32 | 115.2ms |
| `mlp.gate_proj` | **sparse_nvfp4** | 2.46ms | 32 | 78.6ms |
| `mlp.up_proj` | **sparse_nvfp4** | 2.45ms | 32 | 78.3ms |
| `k_proj` | **marlin_nvfp4** | 0.62ms | 8 | 5.0ms |
| `o_proj` | sparse_bf16 | 1.27ms | 8 | 10.2ms |
| `q_proj` | **sparse_nvfp4** | 2.13ms | 8 | 17.1ms |
| `v_proj` | **marlin_nvfp4** | 0.62ms | 8 | 5.0ms |

**Prefill total: 427ms**（vs dense_bf16 973ms → 2.28× faster on linears）

### Decode 阶段 (M=32) — 2种 kernel

| 层类型 | 最优 Kernel | 单层延迟 | 层数 | 小计 |
|--------|------------|---------|------|------|
| `in_proj_a` | **dense_bf16** | 0.04ms | 24 | 1.0ms |
| `in_proj_b` | **dense_bf16** | 0.04ms | 24 | 0.9ms |
| 其余全部10种层 | **marlin_nvfp4** | 0.04-0.05ms | — | 8.4ms |

**Decode per-step: 10.3ms**（vs dense_bf16 13.1ms → 1.27× faster on linears）

---

## 关键发现

1. **9B 是 shape-workload hybrid 的 sweet spot**
   - 维度足够大：sparse/quant kernel 的 compute 收益 > packing overhead
   - 单卡 32GB 放得下：无多卡兼容问题
   - Prefill 大 M 用 sparse 家族，Decode 小 M 用 marlin → 完美互补

2. **小模型 (0.8B/2B) 不适合 hybrid**
   - N=16 的 MLA 层上 sparse overhead > 收益
   - 需要按模型尺寸重新 tuning 策略

3. **4B 处于临界点**：hybrid ≈ dense (0.99x)

4. **NVFP4 量化在 decode 阶段是灾难**
   - dense_nvfp4 / sparse_nvfp4 在 M=32 时比 dense 慢 2-10×
   - 小 batch 下 activation packing overhead 完全碾压计算收益
   - W4A16 (marlin) 是唯一在 decode 可用的量化方案

5. **27B 有 kernel 级阻塞问题**
   - marlin_nvfp4 多卡 forward crash（单层OK，全模型crash）
   - hybrid 双 backend 显存翻倍
   - 需要 kernel 代码修复 + 更轻量的 hybrid 实现

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `scripts/bench_qwen3_5_shape_workload_hybrid.py` | Module 级 hybrid 验证脚本 |
| `scripts/bench_qwen3_5_swh_e2e.py` | 真实 E2E 推理 benchmark |
| `fake/models/qwen3_5_kernels.py` | `shape_workload_hybrid` method 实现 |
| `artifacts/results/benchmarks/hybrid/qwen35_9b_shape_workload_hybrid.csv` | 9B module 级结果 |
| `artifacts/results/benchmarks/module/Qwen3.5-9B/kernel/` | 9B 全 kernel 实测数据 |

---

*生成时间: 2026-06-03 | GPU: NVIDIA RTX 5090 32GB ×8 | CUDA: 12.8 | PyTorch: 2.9.0*
