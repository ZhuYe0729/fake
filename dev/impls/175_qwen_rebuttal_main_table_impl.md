## 2026-10-28 - Qwen双场景15行主表
- 按用户要求建立唯一清晰的Qwen展示入口reviewer2/qwen3_main_table：保留原uniform压缩、INT8/FP8/SmoothQuant、固定阶段baseline，以及HAWQ/DASO各Balanced和Max-speed两行；移除主视图中的DP/validation版本堆叠及诊断控制标签。
- 主Markdown为15行摘要（两个场景Avg及速度），可展开14列完整Table2；另提供有Prefill/P+D分组表头的HTML、未舍入CSV、逐场景配置/来源JSON。Avg分别用四MC和三生成分项未舍入值等权计算，两个场景独立配置。
- PP Balanced的DASO点来自015已验证SQ-INT8选点，PP Max来自016硬件最小延迟incumbent；PD采用048同轮D1/D32。HAWQ Balanced为r=.80，Max为原真实Hessian/ILP在最小可行Linear上限的输出。原较小请求曾不可行的记录原样保存，不将其写成原目标通过。HAWQ端点与048共享硬件点map逐项完全相同，非任意控制点改名。
- HAWQ与DASO的PP Max完整method_map相同，因此共用016同一份质量及速度；SQ-INT8与DASO PP Balanced同理。名称只定义本表报告的两个代表工作点，不称全候选全预算全局最优；HAWQ注明FP4 PTQ适配，多campaign归一化范围明确。
- 195个指标完整（165原指标+30 Avg）；105个质量单元逐一回读原完整任务/生成JSON，136份来源SHA保持，015/016原manifest SHA一致。摘要/完整Markdown270数值格与CSV匹配，HTML15×6/15×14列结构及本地链接通过；按展示精度并列最高值共同加粗。
- P+D速度优先点DASO E2E2.032×、Avg29.951，对HAWQ1.905×、29.753；E2E比值约+6.69%，Avg仅称点估计+0.199，不作显著性或全面最优结论。未新增GPU测量，未修改原封存表/代码/日志/结果；responses/README增加新主表入口。
