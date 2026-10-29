# Rebuttal人工reviewer证据索引

**Qwen主结果优先看这里：[15行双场景主表](reviewer2/qwen3_main_table/README.md) · [分组列HTML](reviewer2/qwen3_main_table/table.html)**。包含全部baseline、HAWQ/DASO各两个报告工作点，以及两场景各自Avg；历史诊断版本不放入该主表。

[其余实验结果清晰版](../REBUTTAL_SELECTED_EVIDENCE.md)：按reviewer问题查看质量预算、多负载、混合调度、内存/切换、建模/成本、剪枝和反约束；来源与协议折叠。

两套原模型也已统一主表展示：[Llama-3.1-8B完整baseline主表](reviewer2/llama31_main_table/README.md) · [Llama-2-7B完整baseline主表](reviewer2/llama2_main_table/README.md)。两个场景各有Avg/速度，HAWQ与DASO各两个代表点。

**原协议下游精度已补齐：[B8/B4完整任务表](reviewer2/qwen_original_quality_run050/README.md) · [分组HTML](reviewer2/qwen_original_quality_run050/table.html)**。BF16、HAWQ/DASO各两个冻结点，92/92任务或分块完成，包含两场景分项与Avg；沿用原完整任务、prompt与生成上限256。速度单独留待空闲时段重测。本轮分数按map/batch测量，多个测速S/O标签共用时明确标注。

**新增核心负载双场景复测：[四组配置速读表](reviewer2/qwen_nearby_dual_scenario_run049/SUMMARY.md) · [完整35行HTML](reviewer2/qwen_nearby_dual_scenario_run049/table.html)**。三种Prefill＋四种P+D，DASO/HAWQ各两个冻结点，64窗实测NLL＋三轮速度已完成；†行仍有速度波动，逐轮配对和原始范围均保留。原协议下游任务分数另见上方B8/B4表，不混作固定S/O任务精度。

**原六负载P+D补充：[九方法完整质量–速度表](reviewer2/qwen_workload_comparison/README.md) · [分组HTML](reviewer2/qwen_workload_comparison/table.html)**。每种负载均有HAWQ/DASO各两个点、简单阶段组合及8bit参考；精度为各负载实测ΔNLL，速度含TTFT/TPOT/E2E/cycle。与新增核心复测分开读取，不混合两个campaign的数值。

[可直接编辑的逐reviewer精简英文回复](../REBUTTAL_RESPONSE_DRAFT.md) · [实际预算与同map速度关联](../response174/budget_speed.csv) · [六负载质量–速度PDF图](reviewer2/qwen_workload_comparison/quality_speed.pdf) · [预注册延迟校验新交付](reviewer5/latency_validation_night161/README.md)。

reviewer 编号按 reference/reviewers.md 中五位人工 reviewer 的出现顺序映射。原有证据包来自 runs/001；新增 8bit 与扩展候选证据来自 runs/002，均经过完整性校验。

| 目录 | 原审稿编号 | 证据主题 |
| --- | --- | --- |
| [reviewer1/AK6a](reviewer1/qwen3_run001/README.md) | AK6a | 质量预算与双表示内存 |
| [reviewer2/Exi5](reviewer2/qwen3_run001/README.md) | Exi5 | HAWQ-v3 与 Qwen3 基线 |
| [reviewer3/LrT2](reviewer3/qwen3_run001/README.md) | LrT2 | Qwen3 泛化、混合精度和建模开销 |
| [reviewer4/QboE](reviewer4/qwen3_run001/README.md) | QboE | 架构补充与搜索成本 |
| [reviewer5/gkP9](reviewer5/qwen3_run001/README.md) | gkP9 | 局部误差输入与质量模型实证 |

[统一完整报告](../runs/001/REPORT.md)；失败尝试与原始日志保留在 runs/001，不混入这些最终 CSV。

## 新增 8bit baseline 与扩展候选（run002）

- [reviewer1 / AK6a](reviewer1/qwen3_run002/README.md)：预算校验、8bit 小误差动作与全局 DP 网格诊断。
- [reviewer2 / Exi5](reviewer2/qwen3_run002/README.md)：SmoothQuant W8A8、FP8、扩展 DASO。
- [reviewer5 / gkP9](reviewer5/qwen3_run002/README.md)：INT8 / FP8 候选空间与同成本输入控制。
- [run002 完整报告](../runs/002/REPORT.md)；001 原证据包保持有效。

## 工作负载覆盖补充（run003）

- [reviewer2](reviewer2/qwen3_run003/README.md)：batch、提示长度、生成长度与按负载重分配。
- [reviewer1](reviewer1/qwen3_run003/README.md)：跨负载实际质量与预算满足情况。
- [reviewer5](reviewer5/qwen3_run003/README.md)：固定迁移、成本变化和真实端到端排序。

## 人工reviewer夜间补充（plan144）

[总体结果与结论边界](../OVERNIGHT_REPORT.md) · [五位人工reviewer concern矩阵](HUMAN_REVIEWER_EVIDENCE.md) · [证据包副本/来源审计](overnight_packaging_audit.json)。004–011八个目录均已封存，共21份人工reviewer证据包，全部GPU作业已结束。

| Reviewer | 已整理的新增证据包 |
| --- | --- |
| AK6a / reviewer1 | [004开销/内存](reviewer1/overnight_run004/README.md)、[005实际质量/验证](reviewer1/overnight_run005/README.md)、[006实际mixed质量](reviewer1/overnight_run006/README.md)、[009强化HAWQ](reviewer1/overnight_run009/README.md)、[010完整任务](reviewer1/overnight_run010/README.md)、[011统一比较](reviewer1/overnight_run011/README.md) |
| Exi5 / reviewer2 | [006真实mixed两轮](reviewer2/overnight_run006/README.md)、[009强化HAWQ](reviewer2/overnight_run009/README.md)、[011统一比较](reviewer2/overnight_run011/README.md) |
| LrT2 / reviewer3 | [004预测/完整成本](reviewer3/overnight_run004/README.md)、[009强化HAWQ](reviewer3/overnight_run009/README.md)、[010完整任务](reviewer3/overnight_run010/README.md)、[011统一比较](reviewer3/overnight_run011/README.md) |
| QboE / reviewer4 | [004真实大shape](reviewer4/overnight_run004/README.md)、[007简单剪枝](reviewer4/overnight_run007/README.md)、[008修正版矢量图](reviewer4/overnight_run008/README.md)、[010完整任务](reviewer4/overnight_run010/README.md) |
| gkP9 / reviewer5 | [004接口/传输](reviewer5/overnight_run004/README.md)、[005反向SLA/数值网格](reviewer5/overnight_run005/README.md)、[008定位/相同输入](reviewer5/overnight_run008/README.md)、[011统一比较](reviewer5/overnight_run011/README.md) |

## 统一Graph运行时与重分析（run015）

W4A16 TPOT诊断后，统一执行路径、新画像和重新分配后的比较见[完整主表](reviewer2/qwen3_table2/README.md)。旧eager结果作为对应运行时的历史证据保留，不能与新Graph数字混排。

[reviewer1最终补充](reviewer1/graph_run015/README.md) · [reviewer2最终补充](reviewer2/graph_run015/README.md) · [reviewer3最终补充](reviewer3/graph_run015/README.md) · [reviewer4最终补充](reviewer4/graph_run015/README.md) · [reviewer5最终补充](reviewer5/graph_run015/README.md)
# 补充实验总清单

[Llama与Qwen：18张按人工reviewer对应的完整计划表](../EXPERIMENT_CHECKLIST.md)，包括当前完成状态、原文数值、待补baseline、优先级与证据来源。


## 本轮三模型补充（plan151）

- [Qwen完整19行汇总](reviewer2/qwen3_table2_run016/README.md)：015/016跨campaign来源分列，DASO-8完整质量已齐。
- [Llama-3.1统一新主表](reviewer2/llama31_8b_instruct_table2/README.md)、[Llama-2统一新主表](reviewer2/llama2_7b_chat_table2/README.md)：两模型15行核心完整质量和三轮速度均已完成并审计。
- [R2同候选与预算配对诊断](reviewer2/matched_allocators_llama_night151/README.md)：12组固定同约束对照，包含未测速和不利点。
- [R3三模型建模诊断](reviewer3/modeling_night151/README.md)：固定LUT+BF16残差，完整质量代理覆盖范围、PDF/SVG与来源SHA。
- [R1 DP网格与反约束复核](reviewer1/solver_grid_run023/README.md)、[R5副本](reviewer5/solver_grid_run023/README.md)：16点DP/MILP和8项inverse检查；仅针对代理。
- [夜间报告与实时进度](../NIGHT151_REPORT.md)：019负载、020剪枝、021/022五预算及测速、025接口、026样本量跟进均已完成。

[32→64 validation窗口跟进（R1/R2/R5）](reviewer2/validation_size_run026/README.md)：固定14个主表incumbent、独立新确认；原封存主表保留，改善与局限单列。

## 本轮最终交付入口

- [六负载完整66行与图](reviewer2/workloads_run019/README.md) · [PDF图](reviewer2/workloads_run019/qwen019_workloads.pdf)
- [Llama-3.1简单剪枝完整对照](reviewer4/pruning_llama31_run020/README.md)
- [两Llama五档预算完整20点](reviewer1/five_budget_llama_night151/README.md)
- [三模型18行内存/cycle](reviewer1/runtime_memory_night151/README.md)
- [768条8bit真实接口](reviewer5/interfaces_run025/README.md)
- [全部结果、限制及保存审计](../NIGHT151_REPORT.md)

并行后续：027/028独立后续工作未纳入本轮016–026完成统计；028实时阶段以[状态与协议](../runs/028/README.md)所指的results/worker.json为准。
