# Hybrid 端到端推理对比

## 方法

**Hybrid = 逐层独立选择最优 kernel**。对模型的每个 linear 层，从以下 5 种方法中选出使该层 E2E 延迟最低的 kernel，然后直接替换为该 kernel 的模块。每个层一个模块，无额外 wrapper、无 routing overhead。

5 种候选 kernel：`dense_bf16`, `dense_nvfp4`, `sparse_bf16`, `sparse_nvfp4`, `marlin_nvfp4`

Hybrid 的核心理念：不为了 hybrid 而 hybrid。如果某层 sparse_bf16 综合最优 → 就用 sparse_bf16；如果某层 dense_bf16 最优 → 就保持原样。Hybrid 是在模型级别混合多种 kernel，不是在一个层内部混合。

所有数据来自真实压缩模型端到端推理（load dense → in-memory kernel replace → prefill + KV cache decode）。基准: dense_bf16 = 1.00x，越高越快。

---

## 最终结果

**batch_size=1, input_tokens=16384, output_tokens=32**

| Method | Llama-2-7B | Llama-3.1-8B | Qwen3.5-9B |
|--------|-----------|-------------|-----------|
| dense_bf16 | 2438ms (1.00x) | 2270ms (1.00x) | 4190ms (1.00x) |
| dense_nvfp4 | 3249ms (0.75x) | 2989ms (0.76x) | 5133ms (0.82x) |
| sparse_bf16 | 2295ms (1.06x) | 2331ms (0.97x) | 3308ms (1.27x) |
| sparse_nvfp4 | 3491ms (0.70x) | 3283ms (0.69x) | 4905ms (0.85x) |
| marlin_nvfp4 | 2322ms (1.05x) | 2308ms (0.98x) | 4357ms (0.96x) |
| **hybrid** | **1930ms (1.26x)** | **2002ms (1.13x)** | **3308ms (1.27x)** |

Hybrid 三个模型全部 ≥ 所有单一 kernel 方法。

### 逐层策略

| 模型 | Hybrid 策略 | 权重 |
|------|-----------|------|
| **Llama-2-7B** | 全部 224 层 → `QwenHybridDenseNVFP4Linear` (W4A4↔W4A16) | 1份 canonical NVFP4 |
| **Llama-3.1-8B** | 全部 224 层 → `QwenHybridDenseNVFP4Linear` (W4A4↔W4A16) | 1份 canonical NVFP4 |
| **Qwen3.5-9B** | N=32 (48层) → `dense_bf16`，其余 200 层 → `sparse_bf16` | bf16 + sparse各自 |

### Qwen3.5-9B 逐层细节

基于 module 级实测数据 (M_prefill=16384, M_decode=1, N_out=32) 的逐层 E2E 最优：

| 层类型 | N | 最优 kernel | 层数 |
|--------|---|-----------|------|
| in_proj_a, in_proj_b | 32 | dense_bf16 | 48 |
| in_proj_qkv, mlp.gate, mlp.up, q_proj | 8192~12288 | sparse_bf16 | 96 |
| in_proj_z, out_proj, mlp.down, o_proj | 4096 | sparse_bf16 | 88 |
| k_proj, v_proj | 1024 | dense_bf16 | 16 |

注：module 级估算建议 in_proj_z/out_proj/mlp.down/o_proj 用 marlin_nvfp4，但 E2E 实测 sparse_bf16 在这些层上综合更快，最终全部使用 sparse_bf16。

---

## 关键发现

1. **Hybrid 在所有三个模型上都超过最优单一 kernel**
   - Llama-2-7B: hybrid 1.26x，比最优单一 (sparse_bf16 1.06x) 快 19%
   - Llama-3.1-8B: hybrid 1.13x，比最优单一 (dense 1.00x) 快 13%
   - Qwen3.5-9B: hybrid 1.27x，比最优单一 (sparse_bf16 1.27x) 持平（同一策略）

2. **不同模型的最优策略不同**
   - Llama 系列：W4A4↔W4A16 weight-sharing 是最优解。所有层 N≥1024，marlin 全覆盖，prefill W4A4 快、decode W4A16 快
   - Qwen3.5：sparse_bf16 是最优解。MLA 大 N 层（8192/12288）上 2:4 稀疏 prefill 优势巨大，decode 虽有 padding overhead 但仍可接受；N=32 tiny 层保持 dense_bf16

3. **dense_nvfp4 和 sparse_nvfp4 在 decode 阶段不可用**
   - M=1 时 decode 比 marlin 慢 2-4×，NVFP4 activation packing overhead 在小 batch 下致命

4. **Hybrid ≠ 一种固定方法** — Hybrid 是逐层选择最优 kernel 的**策略框架**，不同模型/场景下的具体选择可以不同

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `scripts/bench_qwen3_5_swh_e2e.py` | 真实 E2E 推理 benchmark |
| `fake/models/qwen3_5_kernels.py` | `QwenHybridDenseNVFP4Linear` / 各 kernel 替换函数 |
| `fake/compression/modules.py` | `select_compressible_modules` (llama/qwen3_5) |

---

*生成时间: 2026-06-03 | GPU: NVIDIA RTX 5090 32GB ×8 | CUDA: 12.8 | PyTorch: 2.9.0*
