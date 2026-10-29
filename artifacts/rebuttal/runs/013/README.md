# Run013：Qwen3 Table 2 补齐

本轮固定原主表17行、22个唯一method-map、32个policy/scenario，不根据新增分数换策略。原表及生成代码在`reference/table_before/`完整保留。

- `configs/design.json`：冻结配置、质量来源、方法顺序与三轮计时协议。
- `configs/speed_inputs.json`：三轮及所有方法共用的实际输入token，seed144983，B8/S2048。
- `results/tasks/`：3个新增HAWQ Prefill配置的完整WikiText、Winogrande、ARC-E、ARC-C、MMLU。
- `results/generation/`：2个新增HAWQ P+D配置，各CNN/DM1000、DialogSum1500、IWSLT333；保留逐样本输出。
- `results/speed/node0/`：GPU0/CPU0–7/node0，6GiB BF16 KV，O1/O64，3 warmups+8 timed，三轮完整方法顺序及首尾BF16锚点。
- `results/quality_summary.csv`：全部17行的完整任务精度汇总；已有结果按完整map核验继承。
- `results/marlin_repeat_probe.json`、`marlin_model_variation_diagnostic.json`：原生INT4配置输出重复差异的诊断。保留原失败断言和修订源码；不更换kernel或选择有利重复。
- `results/worker.json`：运行状态；全部完成后由`finish_when_done.py`更新用户指定主表并执行审计。

P+D首token在EngineCore输出进入FINAL_ONLY过滤前实际观察；TPOT按同一次`(E2E−TTFT)/63`计算。固定生成长度并忽略EOS，记录输出变体，不把生成token逐字一致作为速度筛选条件。恢复prefill权重的时间另存`restore_ms`与`cycle_ms`，E2E范围沿用原协议。

模型导出使用`/dev/shm/daso_rebuttal_013_20261026`；只清理任务自有、可重建的`model.safetensors`，逐项留下metadata和清理记录。稀疏canonical权重、代码、日志、CSV、结果与文档保留。

原始命令、退出状态与当次Python源码快照分别在`logs/`、`code_versions/`。已封存001–012的manifest SHA在`reference/protected_manifests.json`核验。
