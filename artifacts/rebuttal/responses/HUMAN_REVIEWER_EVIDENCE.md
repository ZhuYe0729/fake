# 五位人工 reviewer 的 concern—证据矩阵

只覆盖reference/reviewers.md中五位人工reviewer；AI Reviewer内容不纳入本轮。

| Reviewer | 核心concern | 已完成证据 | 夜间补充与状态 |
| --- | --- | --- | --- |
| AK6a / reviewer1 | 实际质量预算、异构对照、双表示内存、模型家族 | [003](../runs/003/REPORT.md)16负载；[005](../runs/005/REPORT.md)独立validation/test、8预算点0违约；[004](../runs/004/REPORT.md)GPU/host/pinned成本；[009](../runs/009/REPORT.md)强化HAWQ；[010](../runs/010/REPORT.md)完整任务；[011](../runs/011/REPORT.md)同卡统一比较 | 原预算违约仍保留；验证阶段新增且有成本，预算点包含重复配置；跨任务质量无保证 |
| Exi5 / reviewer2 | 真实混合prefill/decode、单一B/S/O、缺少HAWQ/SQ/FP8 | [002](../runs/002/REPORT.md)真实8bit；003完整B/S/O；[006](../runs/006/REPORT.md)真实chunked scheduler与逐token双kernel路由、实际B/S/O质量；009强化HAWQ | 006八策略两轮54速度组、24实际质量文件完成审计。B16异长case在均值cap=.02下DASO两轮约1.54×；标准B8排名反转保留。新增dual-resident、closed-loop burst，不冒充原实现已有生产continuous serving |
| LrT2 / reviewer3 | Qwen3/混合精度、预测误差、全流程成本、非GEMM | 001–003 Qwen3与真实基线；[004](../runs/004/REPORT.md)留出预测、36控制、12trace、98接口；005额外验证成本；[010](../runs/010/REPORT.md)完整任务；[011](../runs/011/REPORT.md)统一速度 | attention/KV/非Linear维持BF16原路径；整模型残差与准备成本已量化，不把搜索时间冒充从零成本；两个时段性能变化完整保留 |
| QboE / reviewer4 | Figure2、30B、搜索成本、简单剪枝 | [008](../runs/008/REPORT.md)零字体矢量图与视觉审查；005搜索规模；004真实32B维度112forward；[007](../runs/007/REPORT.md)同kernel magnitude/Wanda/SparseGPT；010完整五任务 | 完整30B预训练模型未部署，仅规模部分覆盖；合成shape与重复8B搜索动作不冒充整模型结果 |
| gkP9 / reviewer5 | NAS定位、跨模块格式、更多动作、反向SLA、相同输入 | [NAS定位](../runs/008/NAS_CONTEXT.md)；[相同输入协议与近似](../runs/008/IDENTICAL_INPUTS.md)；004接口/传输；005双向求解、穷举/Pareto和实际SLA；002七候选含INT8/FP8 | 前向/反向代理可解；实际SLA验证prefill全部已测样本满足，P+D仍有违约/不可行。原M64 decode局部近似明确披露，未事后重写拟合 |

## 当前能支持的结论

- 003在实际ΔNLL≤0.01切片中，9个P+D负载有7个由DASO组已测点领先，两组B1负载由BF16领先。它是已测点的事后质量切片，不是预测约束的保证。完整表保留所有不利点。
- 003的prefill重分配14/14点违反预测预算，P+D为1/18。005新增验证规则的8个预算点在独立test中0违约，但含重复选择且不提供形式化泛化保证；完整下游见010。
- 强化HAWQ-FP4具有竞争力，DASO不在所有负载/质量上限占优。006两轮与011统一比较均已完成；标准B8 mixed及中心P+D存在时段排名变化，不用跨卡结果、小幅差距或单次好值声称全面稳定领先。
- 算法搜索的扩展性、真实32B shape可执行性和完整30B模型部署是不同层次的证据。当前未完成30B预训练整模型对照，应明确保留该限制。

004–011均已完成并按reviewer分别封存，共21份证据包；[副本/来源链接审计](overnight_packaging_audit.json)可核验。完整30B预训练模型与生产open-loop硬SLA未完成，不能以shape/搜索代理或均值结果替代。详细结果和答复边界见[夜间汇总](../OVERNIGHT_REPORT.md)。未发送或提交外部reviewer消息。
