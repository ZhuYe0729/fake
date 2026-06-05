# Qwen3.5-2B Manual Hybrid Main Experiment

## 环境

```bash
conda activate cospaq
cd /root/wja/project/my/cospaq/fake
```

本实验不使用超算脚本配置。以下命令默认使用本机 `CUDA_VISIBLE_DEVICES=0`，如需换卡只改这个环境变量。

## 输出目录

- checkpoint：`artifacts/results/main/qwen3_5_2b/checkpoints/<method>/model.pt`
- speed CSV：`artifacts/results/main/qwen3_5_2b/scenes/<scene>/<method>/speed.csv`

## 5 个场景

| Scene | batch | input tokens | output tokens | Hybrid method |
|---|---:|---:|---:|---|
| A_long_context | 1 | 8192 | 512 | `manual_hybrid_m1` |
| B_batched_rag | 4 | 4096 | 512 | `manual_hybrid_m4` |
| C_medium_batch | 8 | 2048 | 256 | `manual_hybrid_m8` |
| D_high_batch_short | 16 | 1024 | 128 | `manual_hybrid_m16` |
| E_long_generation | 1 | 2048 | 1024 | `manual_hybrid_m1` |

## Qwen3.5-2B Linear Shape 与场景策略

本节只记录重复 layer 合并后的 linear 类型，不逐层列出 24 个 layer。`N` 是 linear 输出维度，`K` 是输入维度；GEMM 形状按 `(M,N,K)` 理解，其中 prefill 的 `M=batch*input_tokens`，decode 单步的 `M=batch`。

### 可压缩 Linear 类型

Qwen3.5-2B 语言模型中当前可替换的主要 linear 如下，共 186 个。表中的 `M=*` 列表示该 linear 在对应 GEMM `M` 下的 kernel-level 最快和次快 backend；括号中是相对 `dense_bf16` 的加速比。没有 `*` 的 backend latency 来自 exact shape 的真实 benchmark CSV；带 `*` 的 backend latency 是 speed model 基于真实 benchmark CSV 对未直接覆盖的 exact shape 做出的预测。

| Linear group | Count | N | K | M=1 | M=4 | M=8 | M=16 | M=2048 | M=8192 | M=16384 | 压缩判断 |
|---|---:|---:|---:|---|---|---|---|---|---|---|---|
| `linear_attn.in_proj_a` | 18 | 16 | 2048 | `bf16` (1.00x) | `bf16` (1.00x) | `bf16` (1.00x)<br>`sparse_bf16` (0.36x) | `bf16` (1.00x)<br>`sparse_bf16` (0.37x) | `bf16` (1.00x)<br>`sparse_bf16` (0.32x) | `bf16` (1.00x)<br>`sparse_bf16` (0.50x) | `bf16` (1.00x)<br>`sparse_bf16` (0.84x) | 太小，保留 `bf16` |
| `linear_attn.in_proj_b` | 18 | 16 | 2048 | `bf16` (1.00x) | `bf16` (1.00x) | `bf16` (1.00x)<br>`sparse_bf16` (0.36x) | `bf16` (1.00x)<br>`sparse_bf16` (0.37x) | `bf16` (1.00x)<br>`sparse_bf16` (0.32x) | `bf16` (1.00x)<br>`sparse_bf16` (0.50x) | `bf16` (1.00x)<br>`sparse_bf16` (0.84x) | 太小，保留 `bf16` |
| `linear_attn.in_proj_qkv` | 18 | 6144 | 2048 | `marlin*` (1.70x)<br>`bf16*` (1.00x) | `marlin*` (1.50x)<br>`bf16*` (1.00x) | `marlin*` (1.24x)<br>`bf16*` (1.00x) | `marlin*` (1.30x)<br>`bf16*` (1.00x) | `sparse_nvfp4*` (2.22x)<br>`dense_nvfp4*` (2.02x) | `sparse_nvfp4*` (3.09x)<br>`dense_nvfp4*` (2.45x) | `sparse_nvfp4*` (3.48x)<br>`dense_nvfp4*` (3.39x) | 大矩阵，适合按 M 切换 |
| `linear_attn.in_proj_z` | 18 | 2048 | 2048 | `marlin` (1.45x)<br>`bf16` (1.00x) | `marlin*` (1.14x)<br>`bf16*` (1.00x) | `marlin*` (1.27x)<br>`bf16*` (1.00x) | `marlin` (1.39x)<br>`bf16` (1.00x) | `sparse_bf16*` (1.59x)<br>`sparse_nvfp4*` (1.23x) | `sparse_bf16*` (1.82x)<br>`sparse_nvfp4*` (1.56x) | `sparse_nvfp4` (6.41x)<br>`sparse_bf16` (1.87x) | 大矩阵，适合按 M 切换 |
| `linear_attn.out_proj` | 18 | 2048 | 2048 | `marlin` (1.45x)<br>`bf16` (1.00x) | `marlin*` (1.14x)<br>`bf16*` (1.00x) | `marlin*` (1.27x)<br>`bf16*` (1.00x) | `marlin` (1.39x)<br>`bf16` (1.00x) | `sparse_bf16*` (1.59x)<br>`sparse_nvfp4*` (1.23x) | `sparse_bf16*` (1.82x)<br>`sparse_nvfp4*` (1.56x) | `sparse_nvfp4` (6.41x)<br>`sparse_bf16` (1.87x) | 大矩阵，适合按 M 切换 |
| `mlp.gate_proj` | 24 | 6144 | 2048 | `marlin*` (1.70x)<br>`bf16*` (1.00x) | `marlin*` (1.50x)<br>`bf16*` (1.00x) | `marlin*` (1.24x)<br>`bf16*` (1.00x) | `marlin*` (1.30x)<br>`bf16*` (1.00x) | `sparse_nvfp4*` (2.22x)<br>`dense_nvfp4*` (2.02x) | `sparse_nvfp4*` (3.09x)<br>`dense_nvfp4*` (2.45x) | `sparse_nvfp4*` (3.48x)<br>`dense_nvfp4*` (3.39x) | 大矩阵，适合按 M 切换 |
| `mlp.up_proj` | 24 | 6144 | 2048 | `marlin*` (1.70x)<br>`bf16*` (1.00x) | `marlin*` (1.50x)<br>`bf16*` (1.00x) | `marlin*` (1.24x)<br>`bf16*` (1.00x) | `marlin*` (1.30x)<br>`bf16*` (1.00x) | `sparse_nvfp4*` (2.22x)<br>`dense_nvfp4*` (2.02x) | `sparse_nvfp4*` (3.09x)<br>`dense_nvfp4*` (2.45x) | `sparse_nvfp4*` (3.48x)<br>`dense_nvfp4*` (3.39x) | 大矩阵，适合按 M 切换 |
| `mlp.down_proj` | 24 | 2048 | 6144 | `marlin*` (1.43x)<br>`bf16*` (1.00x) | `marlin*` (1.50x)<br>`bf16*` (1.00x) | `marlin*` (1.48x)<br>`bf16*` (1.00x) | `marlin*` (1.64x)<br>`bf16*` (1.00x) | `sparse_nvfp4*` (1.74x)<br>`sparse_bf16*` (1.64x) | `sparse_nvfp4*` (3.29x)<br>`dense_nvfp4*` (2.17x) | `sparse_nvfp4*` (3.35x)<br>`dense_nvfp4*` (2.80x) | 大矩阵，适合按 M 切换 |
| `self_attn.q_proj` | 6 | 4096 | 2048 | `marlin` (1.15x)<br>`bf16` (1.00x) | `marlin` (1.25x)<br>`bf16` (1.00x) | `bf16` (1.00x)<br>`marlin` (0.78x) | `marlin` (1.10x)<br>`bf16` (1.00x) | `sparse_nvfp4` (4.15x)<br>`dense_nvfp4` (3.44x) | `sparse_nvfp4` (6.51x)<br>`dense_nvfp4` (2.43x) | `dense_nvfp4` (1.95x)<br>`sparse_bf16` (1.86x) | 大矩阵，适合按 M 切换 |
| `self_attn.k_proj` | 6 | 512 | 2048 | `bf16` (1.00x)<br>`marlin` (0.27x) | `bf16*` (1.00x)<br>`marlin*` (0.73x) | `bf16*` (1.00x)<br>`marlin*` (0.64x) | `bf16` (1.00x)<br>`marlin` (0.36x) | `bf16*` (1.00x)<br>`sparse_bf16*` (0.97x) | `sparse_bf16*` (1.51x)<br>`sparse_nvfp4*` (1.10x) | `sparse_nvfp4` (4.26x)<br>`dense_nvfp4` (1.91x) | 中小输出维度，保守选择 |
| `self_attn.v_proj` | 6 | 512 | 2048 | `bf16` (1.00x)<br>`marlin` (0.27x) | `bf16*` (1.00x)<br>`marlin*` (0.73x) | `bf16*` (1.00x)<br>`marlin*` (0.64x) | `bf16` (1.00x)<br>`marlin` (0.36x) | `bf16*` (1.00x)<br>`sparse_bf16*` (0.97x) | `sparse_bf16*` (1.51x)<br>`sparse_nvfp4*` (1.10x) | `sparse_nvfp4` (4.26x)<br>`dense_nvfp4` (1.91x) | 中小输出维度，保守选择 |
| `self_attn.o_proj` | 6 | 2048 | 2048 | `marlin` (1.45x)<br>`bf16` (1.00x) | `marlin*` (1.14x)<br>`bf16*` (1.00x) | `marlin*` (1.27x)<br>`bf16*` (1.00x) | `marlin` (1.39x)<br>`bf16` (1.00x) | `sparse_bf16*` (1.59x)<br>`sparse_nvfp4*` (1.23x) | `sparse_bf16*` (1.82x)<br>`sparse_nvfp4*` (1.56x) | `sparse_nvfp4` (6.41x)<br>`sparse_bf16` (1.87x) | 大矩阵，适合按 M 切换 |

表中缩写：`marlin` 表示 `marlin_nvfp4`，`bf16` 表示 `dense_bf16`。这个表只表示 linear GEMM 的 kernel-level 选择；端到端 decode 已测结果显示 Qwen3.5-2B 的 decode 阶段整体仍是 `dense` 更快，因此最终主实验需要用端到端 benchmark 校验。

### 典型 M 与 Kernel-level 最优策略

下面基于 `5_kernel_comprehensive` 的 standalone GEMM 实测结果，以及该目录下 speed model 对 Qwen3.5-2B exact shape 的预测。它回答的是：如果只看某个 linear 的 GEMM，在给定 `(M,N,K)` 下哪个 kernel 最快。这个结论用于设计候选 hybrid 场景；端到端仍需要实测确认。

典型 M 的含义如下：

| M | 对应推理阶段 | 典型场景 | 设计价值 |
|---:|---|---|---|
| 1 | decode | `batch=1` 每步生成 1 token | 最能体现小 M W-only kernel 的潜在优势 |
| 4 | decode | `batch=4` 每步生成 1 token | 小 batch 服务场景 |
| 8 | decode | `batch=8` 每步生成 1 token | 中等 batch decode，部分 shape 开始回到 BF16 |
| 16 | decode | `batch=16` 每步生成 1 token | 大 batch decode，仍在小 M 范围 |
| 2048 | prefill | `batch=1,input=2048` | 中等长 prompt |
| 8192 | prefill | `batch=1,input=8192` | 长上下文单请求 |
| 16384 | prefill | `batch=4,input=4096` 或 `batch=16,input=1024` | 大 M prefill，最适合展示 WA/shape hybrid |

#### Decode: `M=1`

| 最优 backend | Linear groups | Kernel-level 现象 |
|---|---|---|
| `marlin_nvfp4` | `linear_attn.in_proj_qkv`、`linear_attn.in_proj_z`、`linear_attn.out_proj`、`mlp.gate_proj`、`mlp.up_proj`、`mlp.down_proj`、`self_attn.q_proj`、`self_attn.o_proj` | 对大矩阵通常比 BF16 快，代表性 speedup 约 `1.15x-1.70x` |
| `bf16` | `linear_attn.in_proj_a`、`linear_attn.in_proj_b`、`self_attn.k_proj`、`self_attn.v_proj` | `N=16` 或 `N=512` 太小，压缩 kernel 开销不划算 |

结论：kernel-level 上，`M=1` 适合展示“同一 decode 阶段内按 shape 混合”：大矩阵走 W-only Marlin，小矩阵保持 BF16。

#### Decode: `M=4`

| 最优 backend | Linear groups | Kernel-level 现象 |
|---|---|---|
| `marlin_nvfp4` | `linear_attn.in_proj_qkv`、`linear_attn.in_proj_z`、`linear_attn.out_proj`、`mlp.gate_proj`、`mlp.up_proj`、`mlp.down_proj`、`self_attn.q_proj`、`self_attn.o_proj` | 大矩阵仍倾向 Marlin，代表性 speedup 约 `1.14x-1.50x` |
| `bf16` | `linear_attn.in_proj_a`、`linear_attn.in_proj_b`、`self_attn.k_proj`、`self_attn.v_proj` | 小输出维度继续保持 BF16 |

结论：`M=4` 与 `M=1` 的策略基本一致，但 Marlin 相对 BF16 的优势变小。

#### Decode: `M=8`

| 最优 backend | Linear groups | Kernel-level 现象 |
|---|---|---|
| `marlin_nvfp4` | `linear_attn.in_proj_qkv`、`linear_attn.in_proj_z`、`linear_attn.out_proj`、`mlp.gate_proj`、`mlp.up_proj`、`mlp.down_proj`、`self_attn.o_proj` | 大多数大矩阵仍倾向 Marlin，代表性 speedup 约 `1.24x-1.48x` |
| `bf16` | `linear_attn.in_proj_a`、`linear_attn.in_proj_b`、`self_attn.q_proj`、`self_attn.k_proj`、`self_attn.v_proj` | `q_proj` 在该 M 下回到 BF16，说明 decode 阶段也需要按 shape 决策 |

结论：`M=8` 是展示 shape hybrid 的更明显例子，因为不是所有大矩阵都继续适合 Marlin。

#### Decode: `M=16`

| 最优 backend | Linear groups | Kernel-level 现象 |
|---|---|---|
| `marlin_nvfp4` | `linear_attn.in_proj_qkv`、`linear_attn.in_proj_z`、`linear_attn.out_proj`、`mlp.gate_proj`、`mlp.up_proj`、`mlp.down_proj`、`self_attn.q_proj`、`self_attn.o_proj` | 大矩阵仍多倾向 Marlin，代表性 speedup 约 `1.10x-1.64x` |
| `bf16` | `linear_attn.in_proj_a`、`linear_attn.in_proj_b`、`self_attn.k_proj`、`self_attn.v_proj` | 小/中小输出维度保持 BF16 |

结论：standalone kernel 层面仍支持小 M 使用 W-only，但 `self_attn.q_proj` 的 margin 已经很小，端到端里需要谨慎。

#### Prefill: `M=2048`

| 最优 backend | Linear groups | Kernel-level 现象 |
|---|---|---|
| `sparse_nvfp4` | `linear_attn.in_proj_qkv`、`mlp.gate_proj`、`mlp.up_proj`、`mlp.down_proj`、`self_attn.q_proj` | 大矩阵上 WA + NVFP4 有明显收益，代表性 speedup 约 `1.74x-4.15x` |
| `sparse_bf16` | `linear_attn.in_proj_z`、`linear_attn.out_proj`、`self_attn.o_proj` | 对 `(2048,2048)` 类 shape，sparse BF16 更稳，代表性 speedup 约 `1.59x` |
| `bf16` | `linear_attn.in_proj_a`、`linear_attn.in_proj_b`、`self_attn.k_proj`、`self_attn.v_proj` | 小输出维度保持 BF16 |

结论：`M=2048` 已经能体现 prefill 内部的 shape hybrid：不是统一 sparse NVFP4，而是 `sparse_nvfp4 + sparse_bf16 + bf16` 混合。

#### Prefill: `M=8192`

| 最优 backend | Linear groups | Kernel-level 现象 |
|---|---|---|
| `sparse_nvfp4` | `linear_attn.in_proj_qkv`、`mlp.gate_proj`、`mlp.up_proj`、`mlp.down_proj`、`self_attn.q_proj` | 大矩阵收益更明显，代表性 speedup 约 `3.09x-6.51x` |
| `sparse_bf16` | `linear_attn.in_proj_z`、`linear_attn.out_proj`、`self_attn.k_proj`、`self_attn.v_proj`、`self_attn.o_proj` | `(2048,2048)` 和 `(512,2048)` 类 shape 更适合 sparse BF16，代表性 speedup 约 `1.51x-1.82x` |
| `bf16` | `linear_attn.in_proj_a`、`linear_attn.in_proj_b` | `N=16` 保持 BF16 |

结论：`M=8192` 是展示“prefill 阶段内部按 linear shape 选择不同 WA 方法”的较好场景。

#### Prefill: `M=16384`

| 最优 backend | Linear groups | Kernel-level 现象 |
|---|---|---|
| `sparse_nvfp4` | `linear_attn.in_proj_qkv`、`linear_attn.in_proj_z`、`linear_attn.out_proj`、`mlp.gate_proj`、`mlp.up_proj`、`mlp.down_proj`、`self_attn.k_proj`、`self_attn.v_proj`、`self_attn.o_proj` | 大 M 下 sparse NVFP4 覆盖最多 shape，代表性 speedup 约 `3.35x-6.41x` |
| `dense_nvfp4` | `self_attn.q_proj` | 对 `(4096,2048)` 的 `q_proj`，预测/实测表显示 dense NVFP4 更优，约 `1.95x` |
| `bf16` | `linear_attn.in_proj_a`、`linear_attn.in_proj_b` | 小 N 保持 BF16 |

结论：`M=16384` 是 standalone kernel 层面最能支持 hybrid 的 prefill 场景，因为同一 prefill 内部会出现 `sparse_nvfp4 + dense_nvfp4 + bf16` 三类选择。

### 如何基于这些 M 选择场景

如果只根据 standalone kernel-level 表来设计论文主实验，最有利的场景应该满足两个条件：

- Prefill 的 `M=batch*input_tokens` 足够大，最好是 `8192` 或 `16384`，这样压缩 kernel 的收益更明显。
- Output tokens 不要太长，否则 decode 占主要时间，而当前 Qwen3.5-2B 端到端结果里 decode 用压缩 kernel 反而不占优。

因此更合理的候选场景排序是：

| 优先级 | 场景 | M_prefill | M_decode | 推荐用途 |
|---:|---|---:|---:|---|
| 1 | `batch=16,input=1024,output=1` | 16384 | 16 | 最有利；prefill 大 M，decode 极短 |
| 2 | `batch=4,input=4096,output=1` | 16384 | 4 | 同样大 M prefill，同时展示小 batch decode |
| 3 | `batch=1,input=8192,output=1` | 8192 | 1 | 长上下文单请求，prefill shape hybrid 明显 |
| 4 | `batch=1,input=2048,output=1` | 2048 | 1 | 中等 prefill，收益较小但场景常见 |
| 5 | `batch=16,input=1024,output=128` | 16384 | 16 | decode 较长，hybrid 收益容易被 decode 淹没 |

### 单场景分析：`batch=1,input=8192,output=32`

该场景下：

- Prefill：`M = batch * input_tokens = 8192`
- Decode：每步 `M = batch = 1`
- Decode 重复 32 次，因此不能只看 prefill 最优；decode kernel 的 per-token 开销会被放大 32 倍。

从 kernel-level 表看，`M=8192` 的 prefill 最优大量落在 `sparse_nvfp4/sparse_bf16`，而 `M=1` 的 decode 最优主要是 `marlin_nvfp4` 或 `bf16`。这带来一个实现约束：`sparse_*` 与 `marlin_nvfp4` 不是同一种权重格式，不像 dense NVFP4 的 W4A4/W4A16 可以自然共享同一份 canonical dense NVFP4 权重。因此需要分三种策略讨论。

#### 理论最快：允许同一 linear 保存两套不兼容表示

| Linear group | Prefill, M=8192 | Decode, M=1 | 说明 |
|---|---|---|---|
| `linear_attn.in_proj_a` | `bf16` | `bf16` | 小 N，始终 BF16 |
| `linear_attn.in_proj_b` | `bf16` | `bf16` | 小 N，始终 BF16 |
| `linear_attn.in_proj_qkv` | `sparse_nvfp4` | `marlin_nvfp4` | 理论最快，但需要 sparse 与 dense-NVFP4 两套表示 |
| `linear_attn.in_proj_z` | `sparse_bf16` | `marlin_nvfp4` | 理论最快，但需要 sparse 与 dense-NVFP4 两套表示 |
| `linear_attn.out_proj` | `sparse_bf16` | `marlin_nvfp4` | 理论最快，但需要 sparse 与 dense-NVFP4 两套表示 |
| `mlp.gate_proj` | `sparse_nvfp4` | `marlin_nvfp4` | 理论最快，但需要 sparse 与 dense-NVFP4 两套表示 |
| `mlp.up_proj` | `sparse_nvfp4` | `marlin_nvfp4` | 理论最快，但需要 sparse 与 dense-NVFP4 两套表示 |
| `mlp.down_proj` | `sparse_nvfp4` | `marlin_nvfp4` | 理论最快，但需要 sparse 与 dense-NVFP4 两套表示 |
| `self_attn.q_proj` | `sparse_nvfp4` | `marlin_nvfp4` | 理论最快，但需要 sparse 与 dense-NVFP4 两套表示 |
| `self_attn.k_proj` | `sparse_bf16` | `bf16` | prefill 最优是 sparse，decode 最优是 BF16 |
| `self_attn.v_proj` | `sparse_bf16` | `bf16` | prefill 最优是 sparse，decode 最优是 BF16 |
| `self_attn.o_proj` | `sparse_bf16` | `marlin_nvfp4` | 理论最快，但需要 sparse 与 dense-NVFP4 两套表示 |

Kernel-level linear 总量估计：

- Dense BF16 baseline：`246.93 ms`
- 理论最快两套表示：`119.66 ms`
- 线性层部分约 `2.06x` 加速

这个方案最强，但不是“同权重格式方便切换”的方案。论文中如果使用它，需要明确说明同一 linear 保存了 prefill/decode 两套 backend 表示。

#### 推荐实现方案：dense NVFP4 W4A4/W4A16 切换 + 小层 BF16

如果希望利用目前最清晰的兼容切换，即 W4A4 `dense_nvfp4` prefill 与 W4A16 `marlin_nvfp4` decode 共享 dense NVFP4 canonical 权重，则推荐如下：

| Linear group | Prefill, M=8192 | Decode, M=1 | 推荐原因 |
|---|---|---|---|
| `linear_attn.in_proj_a` | `bf16` | `bf16` | `N=16` 太小，压缩不划算 |
| `linear_attn.in_proj_b` | `bf16` | `bf16` | `N=16` 太小，压缩不划算 |
| `linear_attn.in_proj_qkv` | `dense_nvfp4` | `marlin_nvfp4` | prefill 次优但仍有 `2.45x`，decode 最优 `1.70x`，格式兼容 |
| `linear_attn.in_proj_z` | `dense_nvfp4` | `marlin_nvfp4` | prefill 只有 `1.08x`，但 decode 有 `1.45x` 且可共享 dense NVFP4 权重 |
| `linear_attn.out_proj` | `dense_nvfp4` | `marlin_nvfp4` | 同上 |
| `mlp.gate_proj` | `dense_nvfp4` | `marlin_nvfp4` | prefill 次优但仍有 `2.45x`，decode 最优 `1.70x` |
| `mlp.up_proj` | `dense_nvfp4` | `marlin_nvfp4` | 同上 |
| `mlp.down_proj` | `dense_nvfp4` | `marlin_nvfp4` | prefill 次优 `2.17x`，decode 最优 `1.43x` |
| `self_attn.q_proj` | `dense_nvfp4` | `marlin_nvfp4` | prefill 次优 `2.43x`，decode 最优 `1.15x`，格式兼容 |
| `self_attn.k_proj` | `bf16` | `bf16` | decode 下 Marlin 很慢，prefill sparse 收益不值得引入不兼容切换 |
| `self_attn.v_proj` | `bf16` | `bf16` | 同上 |
| `self_attn.o_proj` | `dense_nvfp4` | `marlin_nvfp4` | prefill 只有 `1.08x`，但 decode 有 `1.45x` 且可共享 dense NVFP4 权重 |

Kernel-level linear 总量估计：

- Dense BF16 baseline：`246.93 ms`
- dense NVFP4 W4A4/W4A16 兼容切换方案：`141.70 ms`
- 线性层部分约 `1.74x` 加速

这是该场景下最适合论文叙述的方案：它同时体现 workload hybrid（prefill/decode 不同 kernel）和 shape hybrid（小层保持 BF16，大层走 NVFP4 切换），并且权重格式兼容。

实现 method：`hybrid_nvfp4_major`。

该 method 只替换以下大 linear：

- `linear_attn.in_proj_qkv`
- `linear_attn.in_proj_z`
- `linear_attn.out_proj`
- `mlp.gate_proj`
- `mlp.up_proj`
- `mlp.down_proj`
- `self_attn.q_proj`
- `self_attn.o_proj`

以下 linear 保持 BF16：

- `linear_attn.in_proj_a`
- `linear_attn.in_proj_b`
- `self_attn.k_proj`
- `self_attn.v_proj`

#### 若完全不允许切换：每个 linear 只能用同一种 backend

如果某个 linear 不能按 prefill/decode 切换，只能固定一种 backend，则 kernel-level 最优会更偏向 Marlin，因为 output token 有 32 个，decode 被重复 32 次：

| Linear group | 单 backend 推荐 |
|---|---|
| `linear_attn.in_proj_a` | `bf16` |
| `linear_attn.in_proj_b` | `bf16` |
| `linear_attn.in_proj_qkv` | `marlin_nvfp4` |
| `linear_attn.in_proj_z` | `marlin_nvfp4` |
| `linear_attn.out_proj` | `marlin_nvfp4` |
| `mlp.gate_proj` | `marlin_nvfp4` |
| `mlp.up_proj` | `marlin_nvfp4` |
| `mlp.down_proj` | `marlin_nvfp4` |
| `self_attn.q_proj` | `marlin_nvfp4` |
| `self_attn.k_proj` | `bf16` |
| `self_attn.v_proj` | `bf16` |
| `self_attn.o_proj` | `marlin_nvfp4` |

Kernel-level linear 总量估计：

- Dense BF16 baseline：`246.93 ms`
- 单 backend per-linear 最优：`202.31 ms`
- 线性层部分约 `1.22x` 加速

这个方案实现最简单，但论文上 hybrid 说服力较弱，因为它主要变成“按 shape 选择 Marlin/BF16”，没有充分体现 prefill 与 decode 的动态切换。

#### 该场景测试命令

目标：测试单一方法与推荐 hybrid 在 `batch=1,input=8192,output=32` 下的真实端到端性能。

结果路径：

- checkpoint：`artifacts/results/main/qwen3_5_2b/checkpoints/<method>/model.pt`
- speed CSV：`artifacts/results/main/qwen3_5_2b/scenes/G_b1_i8192_o32/<method>/speed.csv`

Prepare 推荐 hybrid checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/prepare_qwen3_5_kernel_checkpoint.py \
  --variant 2B \
  --method hybrid_nvfp4_major \
  --dtype bf16 \
  --output artifacts/results/main/qwen3_5_2b/checkpoints/hybrid_nvfp4_major/model.pt
```

如果之前单一压缩方法 checkpoint 已存在，可以跳过下面这段；否则先准备所有单一方法：

```bash
for method in dense_nvfp4 sparse_bf16 sparse_nvfp4 marlin_nvfp4; do
  CUDA_VISIBLE_DEVICES=0 python scripts/prepare_qwen3_5_kernel_checkpoint.py \
    --variant 2B \
    --method "$method" \
    --dtype bf16 \
    --output "artifacts/results/main/qwen3_5_2b/checkpoints/${method}/model.pt"
done
```

Benchmark 所有单一方法和推荐 hybrid：

```bash
run_b1_i8192_o32() {
  local method="$1"
  local extra_checkpoint=()

  if [ "$method" != "dense" ]; then
    extra_checkpoint=(--checkpoint "artifacts/results/main/qwen3_5_2b/checkpoints/${method}/model.pt")
  fi

  CUDA_VISIBLE_DEVICES=0 python scripts/bench_qwen3_5_speed.py \
    --variant 2B \
    --method "$method" \
    "${extra_checkpoint[@]}" \
    --batch-sizes 1 \
    --input-tokens 8192 \
    --output-tokens 32 \
    --warmup 5 \
    --iters 20 \
    --output-csv "artifacts/results/main/qwen3_5_2b/scenes/G_b1_i8192_o32/${method}/speed.csv"
}

for method in dense dense_nvfp4 sparse_bf16 sparse_nvfp4 marlin_nvfp4 hybrid_nvfp4_major; do
  run_b1_i8192_o32 "$method"
done
```

### 端到端已测结果下的保守最强策略

当前 5 个端到端场景已经显示：Qwen3.5-2B 的 decode 阶段 `dense` 全部最快，而 prefill 阶段 `sparse_bf16` 全部最快。因此，如果目标是“主实验尽量不变差”，不要直接采用 standalone 表里 decode 用 `marlin_nvfp4` 的策略，而应使用更保守的 stage-constrained hybrid：

| 阶段 | M | 推荐策略 | 说明 |
|---|---:|---|---|
| Prefill | `M >= 2048` | `in_proj_a/b` 用 `bf16`，其他主要 linear 用 `sparse_bf16` | 来自端到端结果，5 个场景 prefill 都是 `sparse_bf16` 最快 |
| Decode | `M <= 16` | 全部 `bf16` dense | 来自端到端结果，5 个场景 decode 都是 `dense` 最快 |

最有利的补充场景应控制 decode 很短，让 prefill 收益不被长生成淹没：

| Proposed scene | batch | input tokens | output tokens | 推荐 hybrid |
|---|---:|---:|---:|---|
| F_prefill_dominant_short_decode | 16 | 1024 | 1 | prefill: `sparse_bf16` + 小 shape `bf16`；decode: 全 `bf16` |

用已测 D 场景的 stage 数字估计，`B=16,input=1024,output=1` 下：

- Dense 估计：`599.32 ms prefill + 21.61 ms decode = 620.93 ms`
- 新 hybrid 估计：`501.94 ms prefill + 21.61 ms decode = 523.55 ms`
- 预期加速：约 `15.7%`

## Manual Hybrid 方案

这些方案基于 `fake/kernels/cutlass/cutlass_wrapper/artifacts/benchmarks/5_kernel_comprehensive` 的 layer-level GEMM 结果，不基于端到端 CSV 选最快。

### `manual_hybrid_m1`

用于 Scene A/E，decode `M=1`。

Prefill backend：
- `sparse_bf16`：`self_attn.q_proj`、`self_attn.k_proj`、`self_attn.v_proj`、`self_attn.o_proj`、`linear_attn.in_proj_z`、`linear_attn.out_proj`
- `sparse_nvfp4`：`linear_attn.in_proj_qkv`、`mlp.gate_proj`、`mlp.up_proj`、`mlp.down_proj`
- `bf16`：`linear_attn.in_proj_a`、`linear_attn.in_proj_b`

Decode backend：
- `marlin_nvfp4`：除下面 BF16 列表外的主要大 Linear
- `bf16`：`self_attn.k_proj`、`self_attn.v_proj`、`linear_attn.in_proj_a`、`linear_attn.in_proj_b`

### `manual_hybrid_m4`

用于 Scene B，decode `M=4`。

Prefill backend：
- `sparse_nvfp4`：`self_attn.q_proj`、`self_attn.k_proj`、`self_attn.v_proj`、`self_attn.o_proj`、`linear_attn.in_proj_z`、`linear_attn.out_proj`、`mlp.down_proj`
- `dense_nvfp4`：`linear_attn.in_proj_qkv`、`mlp.gate_proj`、`mlp.up_proj`
- `bf16`：`linear_attn.in_proj_a`、`linear_attn.in_proj_b`

Decode backend：
- `marlin_nvfp4`：`self_attn.q_proj`、`self_attn.o_proj`、`linear_attn.in_proj_qkv`、`linear_attn.in_proj_z`、`linear_attn.out_proj`、`mlp.gate_proj`、`mlp.up_proj`
- `bf16`：`self_attn.k_proj`、`self_attn.v_proj`、`linear_attn.in_proj_a`、`linear_attn.in_proj_b`、`mlp.down_proj`

### `manual_hybrid_m8`

用于 Scene C，decode `M=8`。

Prefill backend 与 `manual_hybrid_m4` 相同。

Decode backend：
- `marlin_nvfp4`：`self_attn.q_proj`、`self_attn.o_proj`、`linear_attn.in_proj_z`、`linear_attn.out_proj`
- `bf16`：`self_attn.k_proj`、`self_attn.v_proj`、`linear_attn.in_proj_qkv`、`linear_attn.in_proj_a`、`linear_attn.in_proj_b`、`mlp.gate_proj`、`mlp.up_proj`、`mlp.down_proj`

### `manual_hybrid_m16`

用于 Scene D，decode `M=16`。

Prefill backend 与 `manual_hybrid_m4` 相同。

Decode backend：
- `marlin_nvfp4`：除下面 BF16 列表外的主要大 Linear
- `bf16`：`self_attn.k_proj`、`self_attn.v_proj`、`linear_attn.in_proj_a`、`linear_attn.in_proj_b`

## Prepare Commands

```bash
mkdir -p artifacts/results/main/qwen3_5_2b/checkpoints

for method in dense_nvfp4 sparse_bf16 sparse_nvfp4 marlin_nvfp4 hybrid_nvfp4_major manual_hybrid_m1 manual_hybrid_m4 manual_hybrid_m8 manual_hybrid_m16; do
  CUDA_VISIBLE_DEVICES=0 python scripts/prepare_qwen3_5_kernel_checkpoint.py \
    --variant 2B \
    --method "$method" \
    --dtype bf16 \
    --output "artifacts/results/main/qwen3_5_2b/checkpoints/${method}/model.pt"
done
```

## Benchmark Commands

```bash
METHODS="dense dense_nvfp4 sparse_bf16 sparse_nvfp4 marlin_nvfp4"

run_one() {
  local scene="$1"
  local method="$2"
  local batch="$3"
  local input_tokens="$4"
  local output_tokens="$5"
  local extra_checkpoint=()

  if [ "$method" != "dense" ]; then
    extra_checkpoint=(--checkpoint "artifacts/results/main/qwen3_5_2b/checkpoints/${method}/model.pt")
  fi

  CUDA_VISIBLE_DEVICES=0 python scripts/bench_qwen3_5_speed.py \
    --variant 2B \
    --method "$method" \
    "${extra_checkpoint[@]}" \
    --batch-sizes "$batch" \
    --input-tokens "$input_tokens" \
    --output-tokens "$output_tokens" \
    --warmup 5 \
    --iters 20 \
    --output-csv "artifacts/results/main/qwen3_5_2b/scenes/${scene}/${method}/speed.csv"
}

for method in $METHODS; do
  run_one A_long_context "$method" 1 8192 512
  run_one B_batched_rag "$method" 4 4096 512
  run_one C_medium_batch "$method" 8 2048 256
  run_one D_high_batch_short "$method" 16 1024 128
  run_one E_long_generation "$method" 1 2048 1024
done

run_one A_long_context manual_hybrid_m1 1 8192 512
run_one B_batched_rag manual_hybrid_m4 4 4096 512
run_one C_medium_batch manual_hybrid_m8 8 2048 256
run_one D_high_batch_short manual_hybrid_m16 16 1024 128
run_one E_long_generation manual_hybrid_m1 1 2048 1024
```

## 预期主表读取方式

每个 CSV 中使用以下字段做主表：
- `prefill_ms`
- `decode_total_ms`
- `first_decode_ms`
- `decode_per_token_ms`
- `tokens_per_sec`

主表建议按 scene 展示 6 行方法：
- `dense`
- `dense_nvfp4`
- `sparse_bf16`
- `sparse_nvfp4`
- `marlin_nvfp4`
- 对应 `manual_hybrid_*`

## Module-level Kernel Benchmark

目的：重新测试 5 种 kernel 的真实封装 Linear forward 速度，而不是 GEMM-only 速度。该测试会计入 activation packing、reshape/contiguous、wrapper dispatch 和 output reshape。

输出目录：

```bash
artifacts/results/benchmarks/kernel
```

单卡测试命令：

```bash
python fake/kernels/cutlass/cutlass_wrapper/benchmarks/bench_5_kernel_modules_comprehensive.py \
  --fixed-dim m \
  --gpu 0 \
  --output-dir artifacts/results/benchmarks/kernel \
  --warmup 5 \
  --iters 20
```

3 卡并行测试命令，使用 GPU `1,2,3`：

```bash
bash fake/kernels/cutlass/cutlass_wrapper/benchmarks/run_5_kernel_modules_3gpu.sh
```

可选调整：

```bash
WARMUP=3 ITERS=10 OUTPUT_DIR=artifacts/results/benchmarks/kernel \
  bash fake/kernels/cutlass/cutlass_wrapper/benchmarks/run_5_kernel_modules_3gpu.sh
```

生成文件：

- `artifacts/results/benchmarks/kernel/module_fix_m.csv`
- `artifacts/results/benchmarks/kernel/module_fix_n.csv`
- `artifacts/results/benchmarks/kernel/module_fix_k.csv`
- `artifacts/results/benchmarks/kernel/logs/module_fix_m_gpu1.log`
- `artifacts/results/benchmarks/kernel/logs/module_fix_n_gpu2.log`
- `artifacts/results/benchmarks/kernel/logs/module_fix_k_gpu3.log`

CSV 中 `benchmark_level=module_forward`，表示计时对象是封装后的 `module(x)`。
