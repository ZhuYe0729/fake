## 2026-10-27 - 补齐更完整的FP8基线
- 两Llama原表有plain FP8和SQ-INT8，现补已有实现支持的SQ-FP8，固定原alpha=.5/scales，不调参削弱对照。
- run042/043隔离完整MC/GEN/NLL及五map统一速度，不修改原表或DASO八候选拟合。
- 质量等待037并共享空闲卡；速度等待041后按序，不影响已排正式测量。

## 2026-10-27 - 完整表追加发布器就绪
- night161/code/publish_complete_llama.py在所有原扫点、阶段控制、SQ-FP8完成后创建新的30/29行Llama完整表目录；原15行及27/26行追加表全部数值保留，不覆盖旧目录。
- 两固定阶段控制明确标为控制，不称DASO自动选择；其独立Prefill-only复用对应原uniform结果。SQ-FP8整行使用新MC/GEN及同轮速度，另保留五基线完整对照。
- 两种Avg从未舍入值复算断言；每新增行与campaign/原配置/源审计关联。当前只是发布代码准备好，不代表后续GPU任务已完成。

## 2026-10-27 - 校正报告中的模型专属MC KV说明
- 检查确认实际执行从各自full_task_protocol读取，Llama3.1 MC KV=3GiB、Llama2=10GiB（MHA完整B8需要更大KV），原路径和数值设置始终正确。
- 将新报告/计划中误写为两模型均3GiB的文字改为读取模型协议，不改原已封存报告或实际实验参数。

## 2026-10-27 - 预检报告的真实设备与MC准入字段
- 原benchmark JSON没有顶层gpu字段，报告改为核对每个sample的gpu_before.physical_gpu_index与固定设备、CPU affinity；041同样使用真实样本遥测，不靠不存在的字段。尚未发生新报告失败。
- 新SQ-FP8报告同时核对MC实际last_memory的KV字节数和完整B8/prompt-logprobs数值准入通过记录，避免文字参数正确而实际准入不足。

## 2026-10-27 - 冻结新baseline速度汇总口径
- 042/043两新baseline在GPU输出前固定为各轮同轮BF16归一化后取三轮算术均值，与本轮034/035/036和032/033新点主表一致；原15行不修改，重复范围继续另报。
