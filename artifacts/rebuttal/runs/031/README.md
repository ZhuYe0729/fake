# Qwen3完整候选空间快速对照

- 计划：[plan156](../../../../dev/plans/156_qwen3_full_space_quality_repair_plan.md)。
- 144个融合Linear模块，每模块独立选择8种prefill×8种decode；深度桶/类型仅用于共享校准特征，不绑定模块决策。
- 比较BF16锚定旧代理与signed可加残差代理，同一LUT、精确一选一MILP、同样四个预测目标和共同anchor。主指标沿用论文单批E2E。
- 复用原训练窗口上的130个train/44个policy holdout标签；先选ridge并冻结提案，再使用新128validation/64confirmation，含16个速度输入共208个互不重叠新窗口。
- 本轮仅NLL与速度，未运行新的完整生成/MC。原028/029/030、主表、默认质量模型保持封存。

状态见[worker.json](results/worker.json)，完成后见[REPORT.md](REPORT.md)。全部code、logs、CSV、失败和checkpoint元数据保留；本轮权重可重建并在结束后清理。

首次稀疏导出调用遗漏canonical参数，恢复时补齐016同一SparseGPT权重，未改提案或量化方法；记录见[sparse_export_recovery.json](results/sparse_export_recovery.json)。

## 完成情况

12个validation、6个confirmation、三轮21个正式速度进程已完成并独立核对。旧锚定代理的完整空间ε=.04解通过确认保护值(.02299)，E2E976.87ms、同轮INT8加速1.253×；新残差解更快(959.31ms)，但保护值.04730未通过.04。严格.01尚未改善，两模型选择共同旧incumbent，平均值通过、新64窗口保护值未通过。

本轮证明全空间可以找到快且通过.04确认的点，尚未证明新质量校正可可靠替换原代理。新混合策略完整生成/MC未测。全部细节见[报告](REPORT.md)与[独立核对](results/independent_verification.json)。
