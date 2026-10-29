# Llama3.1-8B-Instruct 消融实验总汇

## 1. 完成情况

按论文当前的核心设计，消融实验已经覆盖完整：

1. **Kernel latency modeling**：direct shape fit、Roofline prior、shape residual correction；
2. **E2E latency composition**：raw module aggregation、PAVA monotonic calibration；
3. **Quality modeling**：local aggregation、structure awareness、phase awareness；
4. **Calibration-set sensitivity**：校准 policy 数量对预测误差与标注成本的影响；
5. **Runtime overlap**：同步阶段边界搬运与异步逐层预取，以及 CUDA-event 直接隐藏比例。

这些实验均以 Llama3.1-8B-Instruct 为展示模型。建模和 runtime 主场景为
prefill-decode，batch size 8、输入长度 2048、输出长度 64。当前没有单独设置
“求解器消融”：求解器执行的是给定预测成本下的确定性约束优化，而不是一个包含可学习
组件的近似模型。`references/wu.tex` 中原先占位的 naive routing/caching 表也没有可靠的
正式测量，因此没有用早期粗测数字填充，runtime 部分只报告已经闭环验证的 overlap。

## 2. 建模消融

### 2.1 Kernel prediction

严格按照未见过的 $(N,K)$ shape group 划分测试集；三个模型使用相同 shape features、
数据划分和超参数。

| Variant | Samples | MAPE ↓ | p90 APE ↓ | Spearman ↑ |
|---|---:|---:|---:|---:|
| Direct shape fit | 4657 | 14.80% | 37.47% | 0.957 |
| + Calibrated Roofline prior | 4657 | 10.78% | 27.92% | 0.962 |
| + Shape residual correction | 4657 | **8.01%** | **22.03%** | **0.972** |

Roofline prior 相比直接拟合将 MAPE 降低 27.2%；shape correction 在此基础上进一步
将 MAPE 降低 25.7%，验证了物理先验和 shape-specific correction 都有独立正收益。

### 2.2 E2E composition

固定最终 refined kernel predictor，只消融从 module/phase aggregation 映射到真实 vLLM
E2E latency 的 policy-level calibration。

| Variant | Holdout MAPE ↓ |
|---|---:|
| Raw E2E aggregation | 18.17% |
| + PAVA monotonic calibration | **8.30%** |

PAVA 将 E2E MAPE 降低 54.3%。由于 E2E holdout policy 数量较少，不将该划分上的
Spearman 作为论文主结论。

### 2.3 Quality prediction

所有模型预测相同 real phase-vLLM teacher-forced $\Delta$NLL，使用54个训练 policy 和
18个严格 holdout policy。

| Variant | Holdout MAE ↓ | RMSE ↓ | Spearman ↑ |
|---|---:|---:|---:|
| Local aggregation | 0.0571 | 0.0711 | 0.920 |
| + Structure awareness | 0.0468 | 0.0568 | 0.944 |
| + Phase awareness | **0.0413** | **0.0529** | **0.963** |

完整模型相对 local aggregation 将 MAE 降低 27.8%，并将排序相关性从0.920提升到
0.963。这里的设计是逐步叠加，而不是分别从同一个 base 独立增加组件。

## 3. 校准样本数量敏感度

固定同一组18-policy holdout，不参与拟合或子集构造。72-policy candidate pool 中每个
子集固定保留5个 dense/uniform anchors，并使用10组固定随机嵌套顺序报告 median 和
IQR。

- 随机子集 median MAE：5个 policy 时为0.0614，45个时为0.0403，54个时为
  0.0408，60个时为0.0392，64个时为0.0394；
- median labeling cost：从5个时3.3 GPU-min增加到64个时63.5 GPU-min；
- 冻结的主实验54-policy模型为0.0413 MAE和48.6 GPU-min；
- 完整72-policy审计达到0.0340 MAE和71.4 GPU-min。

论文主图按既定展示范围画到64个 policy，用于突出实际预算区间中“误差快速下降后边际
收益减弱、标注成本持续增长”的趋势。完整72-policy曲线与所有数值同时保留，不能将
截断图解释为64以后已经严格统计饱和。

## 4. Runtime overlap

正式 E2E 对比使用真实 max-speed `point_011`。该策略的128/128个 Linear module 在
prefill 与 decode 使用不同方法。两个分支只改变 `PHASE_HETERO_SYNC_COPY`，关闭 trace，
同一 pair 在同一 GPU 上交替执行，保留全部7对正式样本。

| Runtime variant | Median E2E latency ↓ |
|---|---:|
| Synchronous phase-boundary copy | 1625.87 ms |
| Asynchronous overlapped prefetch | **1521.73 ms** |

异步 overlap 降低 E2E latency 6.41%（1.068x），并在5/7个 paired comparison 中更快。

更关键的机制诊断使用3次独立 CUDA-event profile：

| Mechanism metric | Median |
|---|---:|
| Total decode-state transfer work | 354.94 ms |
| Exposed copy stall | **1.92 ms** |
| Hidden transfer work | 353.02 ms |
| Hidden ratio | **99.38%** |

`Total transfer work` 是128个逐层 materialization interval 的累计 GPU 时间；
`Exposed copy stall` 是 decode 首次访问权重时在计算流上实际等待 copy event 的累计时间。
隐藏比例不等同于 E2E 加速，因为并行 copy 仍可能竞争 PCIe、显存带宽和调度资源；它
直接说明的是几乎所有搬运没有以显式边界等待的形式落在关键路径上。

## 5. 论文使用建议

- 正文使用根目录 `ablation.tex`；
- Modeling ablation 建议只放一个紧凑表，不使用较单薄的柱状图；
- Calibration sensitivity 使用 `figures/calibration_size_sensitivity.pdf`；
- Runtime 使用同一张表同时给出 E2E 和 direct overlap profile，主结果强调
  **99.4% transfer work hidden**，6.4% E2E reduction 作为补充；
- `figures/modeling_ablation_detailed.*`、完整72-policy曲线和所有原始预测保留给附录或
  审稿补充材料。

## 6. 数据索引

- 汇总数字：`data/key_metrics.csv`
- Kernel/E2E/quality 原始预测与指标：`data/modeling/`
- Calibration repeats、extension NLL、policies与日志：`data/calibration/`
- Runtime 7对E2E样本、日志与3次CUDA profile：`data/runtime/`
- 来源与SHA256：`source_manifest.json`
- 原始复现脚本：`reproduction/`
