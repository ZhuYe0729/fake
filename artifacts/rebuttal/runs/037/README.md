# Llama8bit基线同输入重放核查

等待034/035/036完成，使用相同数值代码和精确export缓存，独立重放两模型plain INT8/SQ-INT8/FP8的原64窗口NLL及每生成任务前128条。保持原batch分组，仅输出路径隔离；不将128条子集与完整基线分数作优劣比较。

[计划](../../../../dev/plans/162_llama_baseline_replay_audit_plan.md) · [状态](results/worker.json) · [固定任务](configs/jobs.json)
