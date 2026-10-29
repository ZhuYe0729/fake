# llama31_8b_instruct：P+D预算与prefill权重补测

来源原run017；独立run034保留原八候选、原模型精度代理和kernel，128模块×64阶段组合自由求解。六标签全部在本轮GPU结果前冻结：wP=1的q=.01/.02/.03/.04，以及q=.03的wP=4/32。BF16预测归零，不声称q是已验证真实ΔNLL保证。

- [计划](../../../../dev/plans/160_llama_budget_prefill_weight_sweep_plan.md)
- [冻结参数](configs/sweep_points.json)
- [CPU配置检查](results/sweep_cpu_verification.json)
- [状态](results/sweep_worker.json)
- 完成后见SWEEP_REPORT.md与results/sweep_final_audit.json。

共11个唯一map；6个新NLL任务、18个新完整生成任务，5份原NLL/15份同map完整生成带SHA复用。完整CNN/DM1000、DialogSum1500、IWSLT333，保持原BOS/PMPD、batch8、256token/EOS和KV。Prefill-only与MC本轮不重做，后续表格复用原独立配置并标明来源。

源权重与canonical SHA、旧17/18/32/33审计及三张原主表SHA已冻结。质量最多3张空闲卡；正式速度GPU在首轮前固定，各点同卡三轮、首尾BF16与同轮INT8，measurement锁不与质量重叠。只清理本轮可重建导出权重，全部代码、日志、CSV、文档及失败保留。
