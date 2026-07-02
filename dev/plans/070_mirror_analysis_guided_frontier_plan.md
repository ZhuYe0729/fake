# 070 MIRROR Analysis-Guided Frontier Plan

## 背景

当前已测点本身离真实 Pareto frontier 可能较远；纯 additive quality model 也暂时不可靠。目标是基于已有 controlled sparse_bf16 诊断和 full validation 结果，人工构造一批更可能接近真实前沿的策略，并直接端到端测试。

## 策略假设

- 以 `dense_bf16` 作为默认运行基底，避免 FP32 dense 与混合 dtype 的额外开销。
- sparse_bf16 只按低 local error 优先加入，避免 speed-greedy 选择，因为 controlled 结果显示 `speed_count_*` 精度不稳定。
- 暂时避免 sparse_nvfp4；已有 full/uniform 结果质量过差，不适合作为前沿候选。
- dense_nvfp4 只用于低 local error 的少量层，作为高精度区的速度补充。
- 构造多档质量约束点：高精度、小幅压缩、中等压缩、高 sparse_bf16、uniform sparse_bf16 参考。

## 步骤

1. 生成 analysis-guided policies。
   - verify: 输出 policy CSV/JSON，包含方法计数和选择规则。
2. 跑真实 forward speed。
   - verify: 输出每个策略的 measured forward_mean_ms 和 AMP-relative speedup。
3. 跑 full quality validation。
   - verify: Chameleon + 8 个 GenImage split 全部完成。
4. 汇总并绘制 measured frontier。
   - verify: 输出 joined CSV、frontier CSV、summary 和 PNG/PDF。

## 产物

- `artifacts/debug/030_mirror_global_pareto/analysis_guided_frontier/`
