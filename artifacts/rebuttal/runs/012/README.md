# Run012 — 排名与约束可靠性复核

对应用户2026-10-26的三个问题：重测DASO/HAWQ速度排名；解释额外validation；区分代理建模误差与同配置重复计时变化，并验证稳定环境与经验时延上界。

六份既有策略、GPU0、真实vLLM。冻结配置见configs/design.json；所有原001–011保持原封存。NUMA双节点copy与模型控制、四个独立本地campaign、校准/test分离的SLA筛选在此独立执行。新质量标签数为0，不将新测速包装为新的质量泛化证明。

这里的validation流程为：训练数据拟合代理并生成候选 → 独立validation实际评估候选 → 按质量/时延可行性筛选并冻结配置 → 另一份test只作最终评价。原005的8个预算点为4档预算×2场景，含重复SQ/BF16选择；不能把8点0超限解释成原代理已经准确或每档都有压缩加速。

[完整报告](REPORT.md) · [独立进程排名](results/ranking.csv) · [全部进程计时](results/process_summary.csv) · [冻结选择](configs/sla_selected.json) · [新时延test审计](results/sla_audit.json)。

执行完成：28个主实验模型进程、4个双驻留进程、4个成功copy进程；40个同输入输出token核验通过。原5%规则在新95% BF16 SLA下11/16超时，按校准尾部换选后0/16；更紧两档在预置小候选池内不可行。四轮DASO5对HAWQ phase-uniform为3快1慢，不能用单轮最佳值声称稳定领先。

质量误差和时延误差都需要验证最终决策，但第三个14%示例来自两批实测，不是直接的预测残差。固定5%不是从误差分布推导的上界。本轮先控制执行环境，再在校准轮预先固定原5%与基于校准最大值的经验上界规则，用未见时延seed检验；目标不可行明确报告。BF16能作为质量回退，但不一定满足比BF16更紧的时延预算。

所有实际子命令及对应源码快照在logs/*.command.json与code_versions/。首批FINAL_ONLY导致的首token观察错误字段排除分解，完整E2E/restore计时保留有效；后续observer版本及断言见代码和observer_correction.json。本轮不产生新的质量标签，不对原质量拟合进行回写。
