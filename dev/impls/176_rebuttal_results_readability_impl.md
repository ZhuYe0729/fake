## 2026-10-28 - 其余实验清晰主表与总览重写
- 按用户认可的Qwen展示方式，新增Llama-3.1/Llama-2各16行主表：全部已完成完整任务的常规baseline、固定阶段组合、HAWQ/DASO各Balanced和Max-speed；双场景各有Avg/速度，完整任务分项折叠，另有CSV/HTML/来源。
- 两模型P+D代表点按已测质量–速度折中展示：Llama3.1为完整空间q=.03/.04、wP=1；Llama2为wP=32/q=.03与wP=1/q=.01。明确是描述性报告工作点，不作共同实际ε或全空间全预算最大值声明。Prefill为独立配置，未把P+D TTFT与另一配置MC拼接。
- HAWQ最小可行端点与原allocation_pool保存的endpoint_hawq_indices逐项匹配；同时核验DASO端点索引和实际被测map。相同Prefill端点共享数值，没有按测量波动制造方法差异。
- 重写REBUTTAL_SELECTED_EVIDENCE.md为10个问题分节、16张表：两Llama主结果、实际质量预算、六负载、mixed、内存/恢复、未知shape预测与部署估计、求解/准备、剪枝、同候选代理、反约束/接口。正文仅用人可读方法名、Balanced/Max-speed；每项先结论再比较表，来源、运行编号与必要边界放折叠说明。Qwen沿用用户已确认主表。
- 旧总览、英文草稿、173来源审计和174审计原样备份于results176/reference；173历史审计明确指向其原文快照。英文回复仅把旧表号引用改为语义锚点，全部英文正文逐行保持一致；responses/README更新清晰入口。
- 验证通过：256份来源SHA；两模型各208指标/32个派生Avg；总览16表94行及全部当前文档22表/161本地链接无错误。Qwen已确认表4份输出SHA与015–018原manifest SHA均不变，原实验结果未修改、无新增GPU测量。审计见results176/audit.json。

## 2026-10-28 - 总览直接加入Qwen并补P+D TTFT
- 按用户反馈将Qwen15行摘要直接嵌入总览第1节，排列在两Llama之前；原先仅提供外链的方式不满足直接查看需求，现已修正。
- 三模型摘要统一为方法＋Prefill Avg/TTFT＋P+D Avg/TTFT/TPOT/E2E七列；Qwen和两Llama的Markdown、HTML列组及生成脚本同步更新，完整任务表不缺列。
- P+D TTFT直接取各行原有pd_ttft_speedup，与独立Prefill TTFT分别展示；三模型CSV逐字节保持不变，没有重新选择配置或更改实验数值。
- 更新前20份文档/脚本/审计快照保留于results176/reference/ttft_followup。更新校验通过：257份来源，47个主表工作点的P+D TTFT均显示并核对；总览17表109行、当前6文档25表/164链接无错误，015–018原manifest不变，无新增GPU测量。
