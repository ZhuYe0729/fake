# Llama2 Pareto 速度-精度建模阶段总结

本文档总结最近几轮围绕 `llama2-7b` 的速度-精度联合建模实验，包括 `prefill_only`、`normal_01`、`normal_02` 三个阶段的主要发现。实验代码和结果主要位于：

- `fake/artifacts/debug/008_llama2_pareto_quality_speed`
- `fake/artifacts/debug/009_llama2_normal01_pareto_handoff`
- `fake/artifacts/debug/010_llama2_normal02_pareto_handoff`

## 1. 总体目标与方法

这一阶段的目标不是直接预测下游任务 accuracy，而是构建一个可解释的 Pareto 选择机制：在给定可接受质量损失预算的情况下，自动为每个 Linear 层选择合适的压缩/推理后端，从而获得尽可能好的真实端到端速度。

采用的优化形式是：

```text
minimize predicted_latency
subject to quality_cost <= budget
```

其中：

- 速度侧使用每个 Linear 层在不同后端下的真实 kernel latency 建模。
- 质量侧使用真实轻量化后收集到的 local error / MSE 类指标构造 proxy。
- 选择结果不是单一方法的 uniform policy，而是 per-linear 的混合策略。
- 验证阶段使用真实模型、真实压缩权重、真实 kernel 做 full-model E2E 和 NLL/ARC 测试。

### 1.1 速度建模方法

速度建模的核心是把一个 full-model policy 的推理时间近似分解为所有 Linear 层在对应后端下的 latency 之和。对每个 Linear 层，根据其形状、所在模块类型以及实际推理场景，候选不同执行后端，例如 `dense_bf16`、`dense_nvfp4`、`marlin_nvfp4`、`dense_nvfp4_prefill_marlin_decode` 等。每个候选后端的 latency 来自真实 kernel benchmark 或已有 warm-E2E-aligned latency 表，而不是简单的理论 FLOPs 估计。

对 `prefill_only` 场景，速度代价主要是：

```text
latency_cost = prefill_latency
```

对 `normal_02` 这类 prefill + decode 场景，速度代价写成：

```text
latency_cost = prefill_latency + output_tokens * decode_latency + conversion_latency
```

其中 conversion latency 用于描述不同权重格式或 backend 切换时的额外开销。最终一个 policy 的 predicted latency 是所有 Linear 层 `latency_cost` 的求和。这个建模方式不能完全等价于真实 E2E，因为 full-model 还包括 attention、KV cache、layernorm、kernel launch、backend dispatch 等非 Linear 开销，但实验表明在合适场景下它能很好地预测不同 policy 的速度排序。

### 1.2 精度/质量建模方法

精度建模没有直接预测 ARC/下游任务 accuracy，而是使用一个质量损失 proxy。原因是下游 accuracy 通常离散、噪声大、样本量敏感，而且不同任务之间并不完全一致。更稳妥的方式是为每个 Linear 层和每种压缩方法估计一个局部损失代价，再用这些局部代价求和，作为整体 policy 的质量损失预算。

当前使用的主要质量 proxy 是基于真实压缩后权重/激活误差得到的 local relative MSE，并结合参数规模和层位置做加权：

```text
quality_cost ~= local_rel_mse * log(numel) * layer/family weight
```

直观理解是：如果某个 Linear 层被压缩后输出误差较大、参数规模较大，或者处在相对敏感的位置，那么它的 quality_cost 更高。一个 policy 的整体质量代价是所有被压缩 Linear 层的 quality_cost 之和。后续用 NLL 和 ARC 等真实指标验证这个 proxy 是否能正确反映质量退化趋势。

### 1.3 速度-精度联合建模方法

联合建模采用约束优化，而不是手工给速度和精度设置加权分数。原因是速度和质量的量纲不同，且质量 proxy 并不等于真实 accuracy，直接加权会引入难以解释的超参数。

因此采用如下形式：

```text
minimize predicted_latency(policy)
subject to quality_cost(policy) <= budget
```

通过改变 budget，可以得到一系列 Pareto 点：

- budget 很小时，策略接近 dense baseline，质量损失最小但速度较慢。
- budget 逐渐增大时，优化器会优先替换那些速度收益大、质量代价低的 Linear 层。
- budget 最大时，得到速度最激进的策略，但质量损失也更大。

这种形式直接回答了当前研究问题：如果愿意牺牲一点质量，应该优先压缩哪些 Linear 层；如果当前速度最优策略质量损失过大，应该优先回退哪些层。最终得到的是一条速度-质量 Pareto frontier，而不是单个固定压缩方案。

### 1.4 指标含义说明

- **E2E latency**：真实 full-model 端到端推理耗时，包括模型所有计算和 runtime overhead，是最终速度验证指标。
- **Predicted latency**：由 per-linear latency 累加得到的预测耗时，用于优化器选择 policy。
- **NLL**：negative log-likelihood，语言模型在验证文本上的负对数似然。NLL 越低表示模型给真实文本分配的概率越高，语言建模质量越好。压缩后 NLL 上升通常表示模型质量退化。
- **PPL**：perplexity，通常由 `exp(NLL)` 得到，也用于衡量语言模型困惑度。PPL 越低越好。
- **ARC acc / acc_norm**：ARC-Challenge 任务上的准确率指标。`acc_norm` 是经过归一化 likelihood 后的准确率，常用于 lm-eval。这里主要作为下游任务 sanity check。
- **Pearson correlation**：衡量两个变量的线性相关性，越接近 `1` 表示线性正相关越强，越接近 `-1` 表示线性负相关越强。
- **Spearman correlation**：衡量排序相关性，不要求线性关系。这里尤其重要，因为优化器更关心不同 policy 的排序是否正确，而不是预测的绝对毫秒或绝对 NLL 是否完全准确。Spearman 为 `1.0` 表示排序完全一致。

## 2. Prefill-Only 阶段结果

`008_llama2_pareto_quality_speed` 首先验证了 `llama2-7b` 在 `prefill_only` 场景下的 Pareto 优化是否成立。

场景：

```text
batch_size = 16
input_tokens = 1024
output_tokens = 0
```

主要结论：

- 速度预测非常可靠：
  - predicted linear latency vs real E2E prefill latency:
  - Pearson `0.9995`
  - Spearman `1.0`
- 质量 proxy 对 NLL 排序非常可靠：
  - quality_cost vs NLL:
  - Pearson `0.9692`
  - Spearman `1.0`
- 对 ARC-Challenge limit=128 也有方向性相关，但任务较噪：
  - quality_cost vs ARC acc_norm:
  - Spearman 约 `-0.79`

代表性 Pareto 点：

| Point | E2E Speedup | NLL Delta | ARC Acc Norm | 说明 |
|---|---:|---:|---:|---|
| 0 | 1.000x | 0.0000 | 0.4609 | dense bf16 baseline |
| 5 | 1.141x | +0.0024 | 0.4609 | 几乎无质量损失 |
| 7 | 1.366x | +0.0268 | 0.4531 | 较好的中间点 |
| 8 | 1.485x | +0.0698 | 0.4531 | 更激进速度点 |
| 10 | 1.661x | +0.5958 | 0.3203 | 质量明显下降 |

这个阶段说明，使用 per-linear constrained optimization 替代 uniform policy 是有效的。它能在速度和质量之间给出连续、可解释的选择，而不是只能选择 all-dense / all-sparse / all-nvfp4。

## 3. Normal-01 阶段发现

`009_llama2_normal01_pareto_handoff` 将场景扩展到：

```text
batch_size = 1
input_tokens = 16384
output_tokens = 32
```

该阶段主要暴露了短 decode 场景的问题：

- prefill 预测仍然可靠。
- 但 total E2E latency 的排序不稳定。
- decode 长度只有 32，mixed-backend 的 dispatch / conversion / KV cache overhead 对总耗时影响很大。
- 某些中间策略在预测上更快，但真实 E2E 反而慢于 dense。

该阶段的结论是：`normal_01` 不适合作为当前方法的主要 positive case。它更像是一个提醒：当 decode 太短时，per-linear latency 的简单加和容易被非 Linear overhead 放大或扭曲。

因此后续转向 `normal_02`，使用更长 decode length 来验证 decode-aware 的策略选择。

## 4. Normal-02 阶段结果

`010_llama2_normal02_pareto_handoff` 聚焦：

```text
batch_size = 1
input_tokens = 16384
output_tokens = 256
```

这个场景更符合长上下文生成，并且 decode 部分足够长，Marlin / hybrid decode 的优势能够体现出来。

### 4.1 速度建模结果

候选方法包括：

- `dense_bf16`
- `dense_nvfp4`
- `marlin_nvfp4`
- `dense_nvfp4_prefill_marlin_decode`

由于 `normal_02` 中 `M=1`，`sparse_bf16` 和 `sparse_nvfp4` 在该候选源中被标记为 unsupported，因此没有进入优化。

经过修复 policy conversion bug 后，稳定 E2E 复测采用 one fresh Python process per repeat，避免单进程多次 decode 带来的显存碎片/OOM。最终对 `point_0/7/9` 做了 3 次独立 repeat：

| Point | 策略概况 | Pred Total | Stable E2E Mean | Std | Speedup |
|---|---|---:|---:|---:|---:|
| 0 | `224 bf16` | 4176.5 ms | 9026.0 ms | 4.5 ms | 1.000x |
| 7 | `153 bf16 + 71 dense_nvfp4->marlin` | 3190.3 ms | 8340.8 ms | 12.4 ms | 1.082x |
| 9 | `128 marlin + 96 dense_nvfp4->marlin` | 2829.1 ms | 7394.2 ms | 5.5 ms | 1.221x |

稳定复测表明：

- predicted ranking 与真实 E2E ranking 一致。
- `point_9` 是当前最佳 operating point。
- `point_9` 相比 dense baseline 获得约 `1.22x` 稳定 E2E speedup。
- 相比已有 single baseline，`point_9` 也优于 all-hybrid / all-marlin 的简单 uniform 策略。

### 4.2 质量建模结果

质量验证覆盖了 `point_0/4/5/6/7/8/9`。其中 `4/5/6/8` 在 32GB full E2E 下 OOM，但 NLL/ARC 可以正常评估，因此保留为质量曲线点。

| Point | Quality Cost | NLL | NLL Delta | ARC Acc | ARC Acc Norm |
|---|---:|---:|---:|---:|---:|
| 0 | 0.0000 | 2.039499 | 0.000000 | 0.40625 | 0.46094 |
| 4 | 0.6481 | 2.064522 | +0.025023 | 0.40625 | 0.46875 |
| 5 | 1.3055 | 2.064560 | +0.025061 | 0.42188 | 0.46094 |
| 6 | 2.5973 | 2.065575 | +0.026076 | 0.40625 | 0.46094 |
| 7 | 5.2974 | 2.068435 | +0.028936 | 0.42969 | 0.46094 |
| 8 | 10.4523 | 2.072573 | +0.033074 | 0.40625 | 0.44531 |
| 9 | 16.5301 | 2.076295 | +0.036796 | 0.40625 | 0.46094 |

质量侧结论：

- quality_cost 与 NLL 单调一致，Spearman `1.0`。
- `point_9` 的 NLL delta 只有 `+0.0368`，属于较小退化。
- ARC-Challenge limit=128 对这些点不敏感，acc_norm 基本不变。
- 当前更适合把 NLL 作为主要质量 proxy，把 ARC 作为 sanity check。

## 5. 当前阶段性结论

目前最重要的 positive result 是 `llama2-7b normal_02`：

- 在真实 full-model E2E 上，`point_9` 达到约 `1.22x` 稳定加速。
- 对应质量损失较小，NLL delta 约 `+0.0368`。
- ARC-Challenge limit=128 没有明显下降。
- 速度排序和质量排序都与 proxy 一致，说明 constrained Pareto optimization 的基本框架成立。

同时也有两个重要经验：

- `normal_01` 这种短 decode 场景容易受到 fixed overhead 和 mixed-backend overhead 影响，不适合作为主要展示场景。
- 对长上下文 decode，E2E 稳定计时最好使用 process-per-repeat，而不是单进程多 iteration，否则容易触发显存碎片和 OOM。

## 6. 后续建议

下一步建议按以下顺序推进：

1. 将 `llama2-7b normal_02` 作为当前主结果固化，用 `point_0/7/9` 展示速度-质量 Pareto。
2. 对 `point_0/7/9` 补更敏感的下游任务，例如 full ARC-Challenge、HellaSwag 或 Winogrande，用于确认 NLL 小幅变化不会造成明显任务退化。
3. 将同样流程迁移到 `llama3.1-8b normal_02`，优先验证 Llama 系列上方法是否可复现。
4. 在 Llama 系列结果稳定后，再扩展到 Qwen3.5-9B。

## 7. 可发给老师的周报总结

本周主要围绕模型压缩策略的速度-精度联合建模做了阶段性实验。我们没有直接预测下游任务 accuracy，而是把问题建模为一个 constrained Pareto optimization：在给定质量损失预算下，为每个 Linear 层选择压缩/推理后端，从而得到一条速度-质量折中边界。实验首先在 Llama2-7B prefill-only 场景验证了基础可行性，per-linear latency 对真实 E2E prefill latency 的 Pearson 相关性达到 0.9995，质量 proxy 对 NLL 的 Pearson 相关性达到 0.9692。随后在 normal_01 中发现短 decode 场景容易受到 mixed-backend overhead 干扰，因此进一步转向 normal_02 长 decode 场景。在 normal_02 中，当前 Pareto 策略能够形成比简单 uniform baseline 更细粒度的折中边界，例如较激进点在保持较小 NLL 上升的同时接近或优于 all-marlin/all-hybrid 这类单一策略，说明 per-linear 约束优化有潜力找到更合理的压缩分配。当前方法仍有不完善之处，包括短 decode 场景的 overhead 建模、部分中间 Pareto 点的 E2E OOM、以及 ARC-Challenge limit=128 对质量差异不够敏感。下一步计划不是单纯扩大测试，而是继续完善速度模型和质量 proxy，补充更敏感的任务验证，并在 Llama3.1-8B 上验证该 Pareto 建模思路是否具有可复现性。
