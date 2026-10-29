# Run006 — 真实混合 token 调度实验

独立dual-resident变体；实际vLLM chunked prefill。原实现限制和新增内存/调度开销分开报告。

[完整报告](REPORT.md) · [最终反向顺序复核CSV](results/serving_counterbalanced_summary.csv) · [全部配置](configs) · [质量与路由原始记录](results)。

原六策略21组速度保留；新增强HAWQ后全部八策略首轮27组、反向顺序27组，最终每非BF16策略合并10次timed burst，BF16合并20次。真实负载质量为24文件/384计分窗口、同16个固定validation文本窗重复用于各case；并非384份独立文本。

运行入口为code/benchmark.py、code/extend_baselines.py与full_quality_queue.py。全部实际命令和对应源码快照见logs/*.command.json、code_versions/。可重建临时权重释放后须按export命令恢复；重放应使用新编号目录，保留本目录封存证据。原001 canonical、002 scales与共享vLLM来源均在命令/metadata中记录。
