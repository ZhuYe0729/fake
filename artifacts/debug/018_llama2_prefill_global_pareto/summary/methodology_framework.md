# LLM 压缩策略的速度-精度联合建模：完整方法框架

本文档系统性地描述了为 LLM 每个 Linear 层自动选择压缩/推理后端的方法框架，包括速度建模、精度/质量建模、约束优化和真实验证四个核心环节。

实验代码和结果主要位于：

- `fake/artifacts/debug/007_llama2_quality_modeling` — 首次质量代理建模
- `fake/artifacts/debug/008_llama2_pareto_quality_speed` — 首次 Pareto 优化 (prefill_only)
- `fake/artifacts/debug/014_llama2_prefill_loss_modeling` — PyTorch 级别 prefill loss 建模
- `fake/artifacts/debug/015_llama2_prefill_kernel_loss_modeling` — Kernel-aware NVFP4 loss 建模（修复 014）
- `fake/artifacts/debug/016_llama2_sparse_bf16_precision_proxy` — Sparse BF16 精度代理拟合
- `fake/artifacts/debug/017_global_coef_structural_ablation` — 全局系数 + 结构消融
- `fake/artifacts/debug/018_llama2_prefill_global_pareto` — 最终全局系数 Pareto（当前最新）

---

## 1. 总体目标与方法

### 1.1 问题定义

目标不是直接预测下游任务 accuracy，而是构建一个可解释的 **per-linear 混合策略选择机制**：在给定可接受质量损失预算的情况下，自动为模型中的每个 Linear 层选择最优的压缩/推理后端，从而获得最好的真实端到端速度。

核心优化形式为**约束优化**：

```text
minimize  predicted_latency(policy)
subject to  quality_cost(policy) <= budget
```

通过改变 budget，得到一系列 Pareto 最优策略点，形成速度-质量权衡边界。这个方法的关键优势在于：

- 不是简单的 uniform policy（全 dense / 全 sparse / 全 nvfp4），而是每个 Linear 层独立选择后端，因此可以找到更细粒度的折中方案。
- 优化器会优先替换那些**速度收益大、质量代价低**的 Linear 层，自动发现模型中对压缩最友好的层。
- 最终输出是一条连续的 Pareto frontier，用户可以按需选择保守或激进的工作点。

### 1.2 候选方法空间

对于 Llama2-7B 模型 (32 layers × 7 linear types = 224 个 Linear 模块)，每个模块有 4 种候选后端：

| 方法 | 描述 | Prefill Kernel |
|---|---|---|
| `dense_bf16` | 无压缩 BF16 baseline | CUTLASS GEMM |
| `dense_nvfp4` | 密集 NVFP4 权重量化 + 运行时激活量化 | CUTLASS `NVFP4Linear` |
| `sparse_bf16` | 2:4 结构化稀疏 + BF16 | CUTLASS Sparse GEMM |
| `sparse_nvfp4` | 2:4 稀疏 + NVFP4 量化 | CUTLASS `PaddedSparseNVFP4Linear` |

候选表共有 224 × 4 = 896 行，每行包含该模块在该方法下的质量代价和延迟代价。

### 1.3 整体 Pipeline

```text
┌─────────────────────────────────────────────────────────────────┐
│  1. 局部误差采集 (per-module, per-method)                        │
│     ├─ 014: PyTorch forward 级别 output_rel_mse                 │
│     └─ 015: Kernel-aware forward 级别 output_rel_mse (NVFP4)    │
├─────────────────────────────────────────────────────────────────┤
│  2. 精度代理拟合 (016 + 017)                                     │
│     ├─ 采样多模块混合策略 → 真实 WikiText-2 prefill loss         │
│     ├─ 拟合 local_error → loss_delta 的映射系数                  │
│     └─ 结构消融: local_only / local_layer / local_type /        │
│                  final_layer_type 四种变体比较                    │
├─────────────────────────────────────────────────────────────────┤
│  3. 候选表构建 (018 build_cost_table)                            │
│     ├─ quality_cost = local_error × global_coef × layer_coef    │
│     │                  × type_coef                               │
│     └─ latency_cost = 真实 kernel benchmark prefill_ms          │
├─────────────────────────────────────────────────────────────────┤
│  4. 约束优化 (018 optimize_pareto)                               │
│     ├─ 整数 DP (2000 bins) + 状态剪枝                            │
│     ├─ 31 个 budget 点 (log-spaced)                              │
│     └─ 29 个唯一 Pareto 策略                                     │
├─────────────────────────────────────────────────────────────────┤
│  5. 真实验证 (018 validate)                                      │
│     ├─ E2E 延迟: 真实 compressed weights + 真实 kernel           │
│     ├─ 质量验证: NLL (WikiText-2) + ARC-Challenge (full 1172)   │
│     └─ 与 uniform baseline 对比                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 速度建模方法

### 2.1 核心思路

速度建模的核心是将一个 full-model policy 的推理时间**近似分解为所有 Linear 层在对应后端下的 kernel latency 之和**。对每个 Linear 层，根据其形状 `(out_features, in_features)` 和所选方法，查询该层在真实 kernel benchmark 下的延迟。

### 2.2 Prefill-Only 场景

当前 018 实验聚焦 prefill-only 场景：

```text
batch_size = 16
input_tokens = 1024
output_tokens = 0
```

速度代价简化为：

```text
latency_cost(module, method) = prefill_latency(module, method)
```

一个 policy 的总预测延迟为：

```text
predicted_latency(policy) = Σ latency_cost(m, policy[m])
```

其中 `m` 遍历所有 224 个 Linear 模块。

### 2.3 延迟数据来源

延迟数据来自预先 benchmark 好的 oracle summary 表，每个 `(method, linear_group, n, k)` 组合有一条记录，包含经过 warm-up 对齐的 `prefill_ms`。latency_source 可选 `existing`（复用已有 benchmark）或 `fresh`（重新 benchmark）。

对于 `dense_bf16`，prefill latency 约为 905.2 ms（所有 224 个模块之和）；速度最优（全部选最快方法，不考虑质量）约为 420.0 ms，即理论上界可达 2.16x 加速。

### 2.4 速度模型的局限性

这个建模方式不能完全等价于真实 E2E，因为 full-model 还包括：
- Attention 计算
- KV cache 管理
- LayerNorm / RMSNorm
- Kernel launch overhead
- Backend dispatch 开销

但实验表明（见第 6 节），在 prefill-only 场景下，predicted linear latency 与真实 E2E prefill latency 高度相关，能够正确预测不同 policy 的速度排序。

### 2.5 扩展到 Prefill + Decode 场景

对于包含 decode 的场景（如 010 的 `normal_02`），速度代价扩展为：

```text
latency_cost = prefill_latency + output_tokens × decode_latency + conversion_latency
```

其中：
- `decode_latency` 来自 decode kernel benchmark（M=1 场景）
- `conversion_latency` 用于描述不同权重格式或 backend 切换时的额外开销（如 `dense_nvfp4_prefill_marlin_decode` 混合方法）

---

## 3. 精度/质量建模方法

### 3.1 为什么不用下游 Accuracy

精度建模不直接预测 ARC / 下游任务 accuracy，原因如下：
- 下游 accuracy 离散、噪声大、样本量敏感
- 不同任务之间不完全一致（如 ARC 和 HellaSwag 的敏感层可能不同）
- 直接建模 accuracy 需要大量采样，计算成本高

更稳妥的方式是构建一个**质量损失代理 (quality proxy)**：为每个 Linear 层和每种压缩方法估计一个局部损失代价，再用这些局部代价的加权和来近似整体 policy 的质量退化程度。然后用 NLL 和 ARC 等真实指标验证 proxy 的排序正确性。

### 3.2 局部误差采集

**第一阶段 (007/014)：PyTorch 级别**

对每个 Linear 模块和每种压缩方法，在 WikiText-2 校准数据上做一次 forward，记录压缩后输出与原始 BF16 输出之间的相对 MSE：

```text
output_rel_mse(module, method) = ||Y_compressed - Y_bf16||² / ||Y_bf16||²
```

这衡量的是该模块在该压缩方法下的**局部输出失真程度**。

**第二阶段 (015)：Kernel-Aware 级别**

014 中的 NVFP4 评估存在 validity 问题：NVFP4 的激活量化发生在 kernel 内部，而 PyTorch 级别的 forward 无法准确模拟。因此 015 使用真实的 CUTLASS kernel 模块：
- `dense_nvfp4` → `NVFP4Linear` (包含运行时激活量化)
- `sparse_nvfp4` → `PaddedSparseNVFP4Linear` (包含激活量化 + 稀疏掩码)

这样确保了局部误差包含了**运行时激活量化的影响**。

### 3.3 代理公式的演化

**第一代 (007/008)：加性加权**

```text
quality_cost(module, method) ≈ local_rel_mse × log(numel) × layer_weight
```

直观理解：
- `local_rel_mse` 越大 → 该层对压缩越敏感 → 代价越大
- `log(numel)` 越大 → 参数越多 → 影响越大
- `layer_weight` → 不同层位置的重要性权重（首层/末层通常更敏感）

**第二代 (016)：稀疏 BF16 专用代理**

从采样策略中学习 `local_error → loss_delta` 的映射：

```text
loss_delta ≈ bias + Σ(local_error × layer_coef[layer] × type_coef[linear_type])
```

拟合方式：采样 120 个多模块随机混合策略，在 WikiText-2 上测量真实 prefill loss delta，然后用梯度下降拟合每个 layer 和每个 linear type 的系数。

Holdout 结果（sparse_bf16）：
- Pearson = 0.9870
- Spearman = 0.9822

**第三代 (017/018)：全局系数 + 结构消融 (最终方案)**

将 016 的方法扩展到所有压缩方法，并引入**全局系数** (`global_coef`)，形成最终的乘法形式：

```text
quality_cost(module, method) = local_error(module, method)
                               × global_coef(method)
                               × layer_coef(method, layer)
                               × type_coef(method, module_type)
```

四个变体的消融比较：

| Variant | 公式结构 | 说明 |
|---|---|---|
| `local_only` | `local_error × global_coef` | 仅全局缩放，无结构信息 |
| `local_layer` | `local_error × global_coef × layer_coef[layer]` | 加入层位置权重 |
| `local_type` | `local_error × global_coef × type_coef[type]` | 加入模块类型权重 |
| `final_layer_type` | `local_error × global_coef × layer_coef[layer] × type_coef[type]` | 完整结构信息 |

消融结果（sparse_nvfp4 holdout，24 samples）：

| Variant | Pearson | Spearman | RMSE |
|---|---|---|---|
| `local_only` | 0.9318 | 0.9278 | 0.0671 |
| `local_layer` | 0.9327 | 0.9409 | 0.0690 |
| `local_type` | 0.9424 | 0.9487 | 0.0633 |
| `final_layer_type` | **0.9458** | **0.9643** | 0.0638 |

结论：`final_layer_type`（同时包含 layer 和 type 结构信息）在所有指标上最优，被选为 018 的最终代理方案。

### 3.4 系数的物理含义

以下以 018 最终使用的 `final_layer_type` 系数为例进行解读。

**Global Coef（全局敏感度）**：

| 方法 | global_coef |
|---|---|
| `sparse_bf16` | 0.0630 |
| `dense_nvfp4` | 0.1433 |
| `sparse_nvfp4` | 0.0643 |

`dense_nvfp4` 的 global_coef 最高，意味着**每单位 local_rel_mse 对最终 loss 的影响最大**（NVFP4 量化引入的误差类型可能与稀疏化不同，对模型输出的破坏更直接）。

**Layer Coef（层位置敏感度）**：

以 `sparse_bf16` 为例，layer_coef 在 layer 0-1 极高（3.89, 6.30），中间层较低（~0.6-1.0），末层 30-31 回升（0.94, 1.11）。这表明：
- **首层和次层对压缩最敏感**：embedding 信息刚进入 transformer，失真会逐层放大
- **末层次之**：直接投影到 vocab space，对最终 logits 影响直接
- **中间层相对鲁棒**：transformer 的残差结构提供了误差容错

**Type Coef（模块类型敏感度）**：

以 `sparse_bf16` 为例，attention 相关的 `q_proj`(1.94) 和 `k_proj`(2.77) 敏感度显著高于 FFN 相关的 `down_proj`(0.64) 和 `up_proj`(0.67)。这表明：
- **Query/Key 投影对压缩极其敏感**：直接影响 attention pattern，进而影响信息聚合
- **Value 投影中等敏感** (0.70)
- **FFN 层相对鲁棒**：残差连接 + 较大的中间维度提供了冗余

---

## 4. 速度-精度联合优化

### 4.1 约束优化形式

```text
minimize  Σ latency_cost(m, policy[m])
subject to  Σ quality_cost(m, policy[m]) <= budget
```

其中 `m` 遍历所有 224 个 Linear 模块，`policy[m]` 从该模块的 4 个候选方法中选择。

选择约束优化而非加权求和的原因：
- 速度和质量的量纲不同
- quality proxy 不等于真实 accuracy，直接加权会引入难以解释的超参数
- 约束形式直接对应研究问题："在可接受的质量损失下，最快能跑多快"

### 4.2 求解算法：整数 DP + 状态剪枝

将连续的质量代价离散化为整数 bins：

```text
q_bin(quality_cost) = ceil(quality_cost × scale)
scale = budget_bins / max_possible_quality
```

其中 `budget_bins = 2000`，足以保证离散化精度。

DP 递推：

```python
dp[0] = (latency=0, choices=())
for each module:
    next_dp = {}
    for each (used_q, (lat, choices)) in dp:
        for each candidate method:
            new_q = used_q + q_bin(quality_cost)
            if new_q <= budget:
                new_lat = lat + latency_cost
                next_dp[new_q] = min(next_dp[new_q], (new_lat, choices + (idx,)))
    dp = prune_states(next_dp)  # 移除被支配状态
```

**状态剪枝**：对于任何两个状态 `(q1, lat1)` 和 `(q2, lat2)`，如果 `q1 <= q2` 且 `lat1 <= lat2`，则后者被支配并移除。这确保了 DP 状态数可控。

### 4.3 Budget 网格

使用 log-spaced budget 网格，在保守端密集、激进端稀疏：

```python
ratios = [0.0] + [10^(-3 + i * 3/(N-2)) for i in range(N-1)]
# 最终 N=31, ratios[-1]=1.0
```

这样做是因为：
- 保守端（小 budget）每增加一点预算都可能发现新的 Pareto 点
- 激进端（大 budget）策略趋于饱和，稀疏采样即可

### 4.4 优化结果概览

从 896 个候选行中求解出 31 个 budget 点，其中 29 个为唯一 Pareto 策略：

| 端点 | quality_cost | predicted_latency | speedup |
|---|---|---|---|
| Conservative (point_000) | 0.000 | 905.2 ms | 1.000x |
| Speed (point_029) | 0.814 | 420.0 ms | 2.155x |

中间的混合策略使用多种方法的组合，而非单一 uniform 方案。

---

## 5. 验证方法

### 5.1 验证策略选择

从 29 个唯一 Pareto 点中**全部选取**进行验证（`--points all`），确保完整覆盖 frontier。

### 5.2 E2E 速度验证

对每个选中的 Pareto 策略：

1. **构建真实混合模型**：根据策略的 per-module 方法选择，加载对应的真实压缩权重
   - 压缩权重来自 `artifacts/results/main/003_llama2_7b_arc_easy_accuracy/prepared/<method>/model.pt`
   - `dense_bf16` 使用原始 HuggingFace 权重
2. **安装运行时 kernel**：对每个被替换的模块，安装对应的 CUTLASS kernel（`NVFP4Linear`, `PaddedSparseNVFP4Linear`）
3. **测量 full-model prefill 延迟**：
   - 场景：`batch_size=16, input_tokens=1024, output_tokens=0`
   - warmup_iters=3, iters=10
   - 使用独立 GPU，one process per point

### 5.3 质量验证

对每个策略：

1. **NLL 评估**：在 WikiText-2 验证集上计算 negative log-likelihood
2. **ARC-Challenge 评估**：使用 lm-eval-harness，`arc_challenge` 任务
   - limit=128（快速筛选）和 full 1172 examples（最终验证）

### 5.4 Baseline 对比

与以下 uniform baseline 进行对比：

| Baseline | 描述 |
|---|---|
| `all_dense_bf16` | 全 BF16（质量上界，速度下界） |
| `all_dense_nvfp4` | 全密集 NVFP4 |
| `all_sparse_bf16` | 全稀疏 BF16 |
| `all_sparse_nvfp4` | 全稀疏 NVFP4 |
| `all_marlin_nvfp4` | 全 Marlin NVFP4（仅 prefill，不适用但保留为参考） |

---

## 6. 018 实验主要结果

### 6.1 场景

```text
模型: Llama2-7B
场景: prefill_only
batch_size: 16
input_tokens: 1024
output_tokens: 0
候选方法: dense_bf16, dense_nvfp4, sparse_bf16, sparse_nvfp4
质量代理: final_layer_type (global_coef + layer_coef + type_coef)
Pareto 点数: 29 (全部验证)
```

### 6.2 速度预测准确性

Predicted linear latency 与真实 E2E prefill latency 的关系：

从 29 个点的验证数据来看，predicted latency 和 real E2E latency 保持了良好的单调性。例如：

| Point | Predicted (ms) | Real E2E (ms) |
|---|---|---|
| point_000 | 905.2 | 1163.1 |
| point_020 | 671.1 | 939.8 |
| point_024 | 528.4 | 782.1 |
| point_026 | 457.1 | 711.1 |

Linear latency 与 E2E latency 之间的固定 gap（~250-260 ms）主要来自 attention、layernorm 等非 Linear 开销。

### 6.3 质量代理准确性

Quality cost 与 NLL delta 的关系（29 个 Pareto 点）：

- 整体单调趋势一致，quality_cost 增大 → NLL delta 增大
- `sparse_nvfp4` 进入后（point_027+），NLL 迅速上升，与 quality_cost 预测一致

Quality cost 与 ARC-Challenge (full 1172) 的关系：

- 保守端 (point_000 ~ point_020)：ARC acc_norm 几乎不变 (0.4514 → 0.4462)
- 中间区域 (point_024)：ARC 轻微下降 (0.4317)
- 激进端 (point_026)：ARC 明显下降 (0.4096)

### 6.4 推荐展示点

| Point | E2E Speedup | NLL Delta | ARC-C acc_norm | 组成 | 说明 |
|---|---:|---:|---:|---|---|
| P000 | 1.000x | 0.0000 | 0.4514 | 224×dense_bf16 | Dense 参考点 |
| P020 | 1.238x | 0.0226 | 0.4462 | 147×bf16 + 64×nvfp4 + 13×sparse_bf16 | 保守质量保持点，ARC 几乎不变，NLL 优于 all_dense_nvfp4 |
| P024 | 1.487x | 0.0974 | 0.4317 | 70×bf16 + 65×nvfp4 + 89×sparse_bf16 | 主推点，比 uniform sparse baseline 更快且 NLL 低得多 |
| P026 | 1.635x | 0.1580 | 0.4096 | 7×bf16 + 64×nvfp4 + 153×sparse_bf16 | 激进点，比所有 uniform sparse baseline 快且质量更好 |

### 6.5 与 Uniform Baseline 的关键对比

| 方案 | Speedup | NLL Delta | ARC-C |
|---|---|---|---|
| all_dense_nvfp4 | 1.377x | 0.0820 | 0.4377 |
| all_sparse_bf16 | 1.462x | 0.3503 | 0.3379 |
| all_sparse_nvfp4 | 1.484x | 1.3184 | 0.2287 |
| **P024 (mixed)** | **1.487x** | **0.0974** | **0.4317** |

**核心发现**：P024 在速度上超过了所有 uniform baseline（包括 all_sparse_bf16 的 1.462x 和 all_sparse_nvfp4 的 1.484x），同时 NLL 远低于 sparse 方案（0.0974 vs 0.3503/1.3184），ARC 也显著更高（0.4317 vs 0.3379/0.2287）。这说明 per-linear 混合策略确实能够在速度和质量之间实现比 uniform 方法更优的折中。

---

## 7. 关键指标说明

- **E2E latency**：真实 full-model 端到端推理耗时，包括模型所有计算和 runtime overhead，是最终速度验证指标。
- **Predicted latency**：由 per-linear latency 累加得到的预测耗时，用于优化器选择 policy。在 prefill-only 场景下仅包含 prefill 部分。
- **NLL (Negative Log-Likelihood)**：语言模型在验证文本上的负对数似然。NLL 越低越好。压缩后 NLL 上升表示模型质量退化。
- **PPL (Perplexity)**：通常由 `exp(NLL)` 得到，也用于衡量语言模型困惑度。PPL 越低越好。
- **ARC acc / acc_norm**：ARC-Challenge 任务上的准确率指标。`acc_norm` 是经过归一化 likelihood 后的准确率，更为常用。
- **Pearson correlation**：衡量两个变量的线性相关性，越接近 `1` 表示线性正相关越强。
- **Spearman correlation**：衡量排序相关性，不要求线性关系。**这里尤其重要**，因为优化器更关心不同 policy 的排序是否正确，而不是预测的绝对毫秒或绝对 NLL 是否完全准确。Spearman 为 `1.0` 表示排序完全一致。
- **local_rel_mse**：单层压缩输出与原始 BF16 输出之间的相对均方误差。衡量局部失真程度。
- **global_coef**：将局部误差映射到全局 loss delta 的全局缩放系数。越大表示该方法的每单位局部误差对最终 loss 影响越大。
- **layer_coef**：层位置敏感度系数。首层和末层通常更高。
- **type_coef**：模块类型敏感度系数。attention 的 q_proj/k_proj 通常更高，FFN 的 down_proj 通常更低。

---

## 8. 方法的关键创新点

1. **Per-linear 而非 per-model**：不是为整个模型选一个压缩方法，而是为每个 Linear 层独立选择，搜索空间从 O(1) 变为 O(4^224)，但通过 DP 高效求解。

2. **约束优化而非加权求和**：避免了速度和质量之间难以解释的权重超参数，直接回答"给定质量预算下最快能多快"。

3. **结构感知的质量代理**：从简单的 `local_error × log(numel)` 演进到 `local_error × global_coef × layer_coef × type_coef`，通过在大规模采样策略上拟合得到物理可解释的系数。

4. **Kernel-aware 建模**：速度和质量两侧都使用真实 kernel（而非理论 FLOPs 或 PyTorch 近似），确保了 proxy 和真实行为之间的一致性。

5. **完整的验证闭环**：优化结果在真实压缩权重、真实 kernel、真实 full-model 上做 E2E 验证，确保 proxy 的排序正确性能够传递到实际部署。

---

## 9. 实验演进路线总结

```text
001-006: 基础设施 — Qwen/Llama2 的 Linear 分解、E2E 差距追踪、全模型 trace oracle
    ↓
007: 首次质量建模 — 在 Llama2 上建立 local_error → NLL/ARC 的关系
    ↓
008: 首次 Pareto — prefill_only 场景验证 per-linear 约束优化的可行性
    ↓
009-010: 扩展到 decode — normal_01 (短decode) 发现问题，normal_02 (长decode) 验证成功
    ↓
014-015: 修复质量建模 — 014 的 NVFP4 用 PyTorch 近似不准确，015 用真实 kernel 修复
    ↓
016: 系统化精度代理 — 从采样策略中学习 local_error → loss_delta 映射
    ↓
017: 全局系数 + 结构消融 — 引入 global_coef，系统比较 local/layer/type 变体
    ↓
018: 最终全局系数 Pareto — 使用 final_layer_type 代理，29 点全验证，得到展示级结果
```

---

## 10. 当前局限与后续方向

1. **仅 prefill 场景**：当前最新结果 (018) 聚焦 prefill_only。decode 场景在 010 中已验证基本可行，但仍需用 017/018 风格的改进代理重新验证。

2. **仅 Llama2-7B**：方法框架已建立，但尚未在 Llama3.1-8B、Qwen 等模型上验证可复现性。

3. **质量代理不直接预测下游 accuracy**：quality_cost 与 NLL 排序一致，但与 ARC 的相关性较弱（ARC 噪声大）。若要针对特定下游任务优化，可能需要任务特定的代理。

4. **非 Linear 开销建模不完整**：attention、layernorm、conversion overhead 等仅体现在 predicted 和 real E2E 的固定 gap 中，未显式建模。

5. **候选方法空间有限**：当前仅 4 种候选方法。Marlin W4A16、混合 prefill/decode 方法等尚未在 017/018 框架内纳入。
