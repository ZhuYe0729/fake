# 050 — Qwen原协议下游精度，速度暂缓

本轮按用户确认的**原精度协议**测试DASO与HAWQ-V3各两个冻结工作点，BF16作参照。当前只运行精度；GPU运行耗时不作为速度证据，速度留待空闲时段独立重测。

[精度表与分项Avg](../../responses/reviewer2/qwen_original_quality_run050/README.md) · [分组HTML](../../responses/reviewer2/qwen_original_quality_run050/table.html) · [实时队列状态](results/worker.json) · [冻结任务设计](configs/design.json)

**已完成92/92个任务或分块。** 共62个独立完整任务分数，展示10行方法/batch及两场景Avg。最终逐样本重算与来源核对见[交付审计](../../responses/reviewer2/qwen_original_quality_run050/delivery_audit.json)；本轮没有新增速度测量。

| 场景 | 完整任务 | 原协议 | 独立测量 |
| --- | --- | --- | --- |
| Prefill-only | ARC-C 1172；ARC-E 2376；Winogrande 1267；MMLU 14042 | 0-shot，原likelihood模板；ARC-C acc_norm，其余acc；上下文2048 | B8/B4各四个独立map，32项完整任务 |
| Prefill+Decode | CNN/DM 1000；DialogSum 1500；IWSLT 333 | 原输入token、无thinking Qwen prompt、greedy、上限256、正常EOS；前两项ROUGE-L，IWSLT完整语料SacreBLEU | B8/B4各五个map；60个对齐分块，最终汇总30项完整任务 |

原生成输入限制3840 token（本地文件实际最长2355），不是测速的固定S=2048/4096；输出也不强制64/128。相同map、batch与原任务输入只测一次；质量可供多个测速S/O行共用，须明确标为原协议任务分数。逐B/S/O NLL另见[049](../049/README.md)。本轮不将旧主表Avg填为新测值。

原主表MC使用B=4、生成使用B=8。本轮将两类任务都分别跑B=8/4，其余任务、prompt及生成上限沿用原设置；不同batch的分数单独保留。

总计92个任务/分块，对应150856个选择题样本与28330个生成样本；展示10行方法/batch，每个场景分别计算完整分项的等权Avg。PP两方法的速度优先端点实际是同一map，复用一次测量。使用049八个权重文件的硬链接，无再次导出；[权重复用身份](reference/checkpoint_reuse.json)与[代码SHA](configs/protocol_sha256.json)记录来源。

空闲GPU逐卡加锁，一卡一个任务，最多四卡并行；子进程核对Torch/NVML GPU UUID。本轮不降低batch、不删不利样本。所有失败尝试和部分输出保留在logs及results/attempts，只有完成标记对应的输出进入最终汇总。

复现锚点已通过：BF16 B8 IWSLT的333条生成token与048逐条一致，BLEU=40.131949202528396；BF16 B4 ARC-E=83.45959595959596%，与原主表一致。首次vLLM UUID设备编号兼容性失败发生在评测前，日志和修复前代码均保留。

执行入口为`code/driver.py`，汇总入口为`../../results179/build_quality.py`。本地冻结数据和缓存全部离线使用，新增Triton/CUDA缓存写入本目录extensions，不清理既有文件。

[开发计划179](../../../../dev/plans/179_qwen_original_quality_first_plan.md) · [开发记录179](../../../../dev/impls/179_qwen_original_quality_first_impl.md)
