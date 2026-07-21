# Llama2-7B-chat canonical prefill Pareto：复盘与防回归指南

## 1. 最终有效结论

本实验的有效实验契约是：Llama2-7B-chat、prefill-only（batch 8、input
2048）、vLLM `phase_hetero_mytest` runtime、真实 vLLM NLL（100 个固定
WikiText block）、五次 loaded-runtime 延迟中位数，以及相同 runtime 的
lm-eval 下游评测。所有 sparse 权重均来自离线生成并验证的 canonical
states，不能在导出时重新执行 `--prune`。

在此契约下，mixed Pareto 曲线已经覆盖了 uniform 方法的重要 trade-off：

| 对照 | ours | 结论 |
|---|---|---|
| Marlin-NVFP4：1.017x、ΔNLL 0.0259 | p010：1.141x、ΔNLL 0.0109 | NLL 和速度均更好 |
| sparse-BF16：1.533x、ΔNLL 0.3457 | p015：1.637x、ΔNLL 0.0279 | 明显更好 |
| dense-NVFP4：1.708x、ΔNLL 0.0421 | p016：1.702x、ΔNLL 0.0396；p017：1.741x、ΔNLL 0.0428 | 分别是质量侧和速度侧的紧邻候选 |
| sparse-NVFP4：1.856x、ΔNLL 1.0171 | p020：1.912x、ΔNLL 0.1611 | 明显更好 |

下游结果不是每项指标都严格支配。例如 p017 的 ARC-Challenge 为 0.4369，
高于 dense-NVFP4 的 0.4317，但其 MMLU 为 0.4435，略低于 0.4494。因此
论文应写成“better trade-off / covers the uniform frontier”，而不是声称
对每一个数据集严格支配。

最终表和图：

- `llama2_7b_chat/pareto/paper/summary.md`
- `llama2_7b_chat/pareto/paper/all_methods_measured.csv`
- `llama2_7b_chat/pareto/paper/pareto_speed_vs_*.png`

## 2. 这次哪些历史结果无效，为什么

### 2.1 直接 `--prune` 不是 canonical sparse 方法

早期 phase exporter 在遇到 sparse policy 时临时追加 `--prune`。它产生的
稀疏权重没有经过本实验要求的离线校准流程；不同导出、不同 policy 之间的
稀疏权重语义也不能保证一致。其典型症状是 uniform sparse-NVFP4 的 ΔNLL
被测为约 6.12，而 canonical sparse-NVFP4 的真实值为 1.0171。

这是**实现错误**，不是模型能力不足。直接 prune 得到的精度训练样本、下游
分数和由其导出的 Pareto 图均不可与 canonical 结果混用。

修复方式：每个线性模块预生成并保存两套离线 state：

- `canonical/prepared/sparse_bf16/model.pt`
- `canonical/prepared/sparse_nvfp4/model.pt`

phase exporter 通过 `--canonical-sparse-bf16-state` 和
`--canonical-sparse-nvfp4-state` 按模块惰性读取对应权重；canonical 模式下
显式禁止 `--prune`。`canonical/verification.json` 必须确认 224 个 linear
weights 均可读取。

### 2.2 runtime 不统一会污染速度和精度比较

历史 uniform 测量使用其专用 uniform 接口，ours 使用 phase runtime。即使
kernel 理论上相近，不同 runner 的模型加载、scheduler、配置和 cache 行为仍
会造成可见偏差：本轮 phase dense-BF16 为 1135.0 ms，而旧 uniform dense-BF16
为 1079.5 ms，约差 5%。这足以改变 dense-NVFP4 邻域是否“覆盖”的判断。

修复方式：uniform 也导出为 all-one-method phase policy，并用同一
`benchmark_phase_baseline_one.py`、同样 warmup、五次测量。NLL 同样通过
phase runtime 再测；本轮 uniform phase NLL 与对应 canonical 质量样本一致。

### 2.3 质量模型的监督标签必须等于最终部署口径

早期质量模型拟合的是旧 runtime 或 direct-prune 的 NLL，而最后报告的是
canonical phase-vLLM NLL。这不是同一个 target，哪怕特征和公式正确，校准
系数的尺度也会错位，导致 solver 选出不合适的点。

有效模型使用同一固定 WikiText block、canonical policy 导出和 phase runtime
测得的 NLL。模块局部误差与 policy 全局 NLL 来自同一套 sparse states。

## 3. 最终建模与求解合同

### 3.1 精度模型

质量代理仍是原先自然的“local + global”加性模型：从逐模块校准误差构造
method/module local features，再使用 policy 级 NLL 拟合共享全局项。它区分
dense、sparse、NVFP4、W4 等方法，不强行假设 sparse 与 quantization 的损失
相同；区别来自其各自真实 canonical local error，而不是人为为某个 uniform
方法添加特例。

本轮 canonical 重校准的 holdout MAE 为 0.0894、RMSE 为 0.1048、Spearman
为 0.7523。实测 Pareto 点还显示模型在中高压缩端普遍保守：p024 预测 ΔNLL
0.9893、实测 0.7408。该偏差应在论文中说明为 calibration residual，不应把
预测值当成最终结果；图和表优先放实测数值。

### 3.2 速度模型

速度仍使用 `KernelLatencyPredictor`：CUTLASS kernel 层面的 roofline/shape
模型加上校准因子，按每层 method 预测 latency；solver 对各层/模块的候选
method 做离散动态规划，在质量约束下最小化预测 prefill latency。它不是查表
选择策略。

速度模型用于**候选搜索**；最终论文速度必须用实际 vLLM E2E prefill 结果
闭环。NLL 与速度都只测代表点即可，但 dense-NVFP4 周围必须额外加密采样，
否则会错过 p016/p017 这种关键近邻点。

### 3.3 求解与验证顺序

1. 拟合 canonical NLL quality proxy，并保留 holdout report。
2. 用速度代理 + quality constraint 求 20--30 个离散 Pareto 点。
3. 先实测稀疏代表点和 uniform endpoints 的 NLL/五次速度。
4. 根据实测速度，在每个 uniform baseline 邻域补 2--4 个 solver 点。
5. 选覆盖低、中、高压缩区间的约 6 个 mixed 点做完整下游任务；uniform
   baseline 也必须用 canonical/phase runtime 补齐。
6. 绘图和论文表只读闭环后的 `paper/` CSV，不直接读预测 CSV。

## 4. 已观察到的问题、根因与处理办法

| 问题 | 根因 | 正确处理 |
|---|---|---|
| sparse uniform NLL 异常巨大 | direct `--prune` 替代了 calibrated pruning | canonical states，禁止 `--prune` |
| ours/uniform 曲线位置不可信 | runner/runtime 不同 | all methods 使用 phase runtime、同 benchmark |
| 多 GPU 导出卡住 | PyTorch extension build lock 残留/并发编译 | 先单 GPU prewarm；确认没有活跃编译后才清理 stale lock |
| 临时 checkpoint 占用大量磁盘 | 同时导出多个 4--13GB 模型 | 每个点完成 NLL/测速后立即删 `/tmp` checkpoint；持久化 JSON/CSV 即可 |
| MMLU 个别任务尝试访问网络 | 数据集缓存尚未完整或并行首次初始化竞争 | 每任务结果持久化并可只重试缺失任务；先让一个 run 填充 cache；失败时不要重跑已完成四项 |
| 速度偶发波动 | GPU 上外部作业或首次运行效应 | warmup + 五次运行取中位数；记录每次 JSON；避开有外部占用的卡 |

## 5. 新模型的硬性 preflight checklist

在启动大规模校准或下游评测前，必须逐项通过：

- [ ] 固定模型 revision、tokenizer、场景参数、policy schema，并写入 provenance。
- [ ] 为每个 sparse method 生成 canonical state，记录 hash/模块数，并验证所有
      linear weights 能被 phase exporter 读取。
- [ ] phase exporter 的 canonical 模式拒绝 `--prune`；导出的 provenance 中记录
      每个模块的 source。
- [ ] 编译/prewarm 所有目标 CUTLASS extension，确认没有活跃 compile process 或
      stale lock 后再做并发导出。
- [ ] 用 phase runtime 跑 dense BF16、所有 uniform compression 和至少一个 mixed
      smoke test；确认实际 activation quantization 和 policy trace 正确。
- [ ] 质量训练集、holdout、solver validation 的 NLL 必须来自同一 tokenizer、固定
      samples、canonical weights 和 runtime。
- [ ] 速度 uniform 与 ours 使用同一 runner、同一 warmup/repeat 数、相同 GPU 空闲
      条件；报告 raw five runs 与 median。
- [ ] 在每个强 uniform baseline 的预测邻域至少实测两个 mixed 点。
- [ ] 下游评测写入单任务可恢复 JSON，任务失败时仅重试缺失 task。
- [ ] 论文图只使用带 `measured_*` 或 `actual_*` 字段的闭环数据；预测值只用于
      diagnostics。

## 6. 可复用入口与关键产物

| 用途 | 入口/产物 |
|---|---|
| canonical states | `scripts/prepare_canonical_sparse.py`、`canonical/` |
| canonical local errors | `scripts/collect_canonical_sparse_local_errors.py`、`local_errors/` |
| NLL 拟合 | `scripts/fit_and_report.py`、`reports/quality/` |
| solver | `scripts/solve_canonical_pareto.py`、`pareto/pareto_points.csv` |
| 点验证 | `scripts/validate_canonical_pareto_point.py`、`pareto/validation/nll/`、`speed/` |
| 下游任务 | `046.../scripts/evaluate_pareto_tasks.py`（显式 canonical states） |
| 最终报告 | `scripts/build_pareto_validation_report.py`、`pareto/paper/` |

## 7. 不应从本轮推导出的结论

- 不应以单个下游指标替代 NLL quality constraint；NLL 是自然且稳定的 solver
  代理，下游任务用于验证 Pareto 趋势与论文展示。
- 不应承诺所有 mixed 点都逐数据集优于每个 uniform 方法；合理目标是以一组点
  覆盖速度--质量 frontier。
- 不应把 direct-prune、旧 runner 或旧 quality 模型的数字与本目录 canonical
  结果放在同一张表或图中。
