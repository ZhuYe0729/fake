# 114 Llama2 canonical Pareto postmortem

## 2026-07-17 - Postmortem recorded
- 开发目的：将本轮 Llama2 canonical Pareto 的有效结论、失效口径与防回归检查固化下来。
- 修改内容：新增复盘与复现实验清单，覆盖 canonical sparse 权重、统一 phase runtime、质量模型标签、速度闭环、下游任务续跑与 artifact 隔离。
- 影响文件：`artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/POSTMORTEM.md`。
- 后续注意：任何新模型在 solver 前必须完成文档中的硬性 gate；不得复用 direct-prune 或跨 runtime 的历史数字。
