# qwen3_8b固定池新窗口质量审计

12套固定map，新validation64与confirmation64。排除旧校准和完整旧validation/test的共享16token片段。公开train语料，非外部新语料；只复核NLL预算，不新增下游任务/速度。等待043完成后启动。

[固定设计](configs/design.json) · [数据排除](reference/data_purge.json) · [计划](../../../../dev/plans/168_fixed_pool_fresh_quality_audit_plan.md)
