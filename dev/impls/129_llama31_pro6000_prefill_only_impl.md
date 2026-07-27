# 129 Llama3.1-8B-Instruct / Pro 6000 / Prefill-only implementation

## 2026-07-22 - 完成独立 prefill-only 论文实验

- 在 067 建立了完整可恢复流程，复制并复验 066 canonical，同时冻结 72 个 Llama3 prefill-only policy 和确定性 WikiText 样本。
- 重新运行 local error、72-policy/100-block NLL、质量拟合、四个精确 shape 的 Pro 6000 profile、24 点 Pareto 求解及 29 点真实闭环。
- 将正式测速修正为每个 policy 单进程内 1 warmup + 5 measured，并验证相同 PID、单 GPU UUID、pure-prefill phase trace。
- 完成 11 个点的 WikiText、WinoGrande、ARC-Easy、ARC-Challenge、MMLU 全量评测，共 55 个正式结果。
- 生成 29 行最终表、六张 Pareto 图；`validate all`、compileall 和 diff check 均通过。
- 所有实验代码和产物位于 `artifacts/debug/067_llama31_pro6000_prefill_only`，旧 debug 目录未修改。
