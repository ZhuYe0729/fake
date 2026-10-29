# Qwen3 工作负载覆盖补充证据

9组P+D与7组prefill-only，覆盖B/S/O变化；包含真实8bit、phase-uniform、固定策略迁移与按负载重分配的DASO/HAWQ。

[完整报告](REPORT.md) · [完整数表](summary_final.csv) · [配置](design.json) · [审计](final_audit.json)。

最终速度统一在GPU0；初始GPU3因外部占卡留下的部分计时仅作diagnostics，见[协议修订](measurement_card_change.json)。

按实际NLL与绝对延迟比较，所有预先指定场景保留。冻结质量模型跨长度迁移；同步同质batch，不主张已验证mixed continuous batching或严格质量保证。
