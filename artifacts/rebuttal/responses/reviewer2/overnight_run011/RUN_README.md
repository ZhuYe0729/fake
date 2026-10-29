# Run011 — 验证后策略与强化HAWQ统一测速

[完整报告](REPORT.md) · [全部43点](results/summary_final.csv) · [预定比较池](configs/candidate_pool.json) · [继承质量来源](reference/inherited_quality.json)。

009全部35点与005全部8个质量预算验证选择点，共19份map、23份质量文件；速度统一GPU0、B8/S2048/O1或64、2 warmups+5 timed、6GiB KV，另有BF16终点锚。没有新增质量标签或改变选择规则。

首轮25速度文件完整保留为diagnostic；待006等其他GPU实验全部结束，再以全部19map反向顺序复测，最终CSV只用后一轮单GPU作业环境。配置与质量不变，不依据输赢增加轮数。首轮报告/CSV/图在results/diagnostics_first_campaign/。

入口为code/setup_run.py、queue_speed.py、write_report.py；所有实际子命令及其源码版本在logs/*.command.json和code_versions/。重放应使用新编号目录，先恢复原canonical/scales和已释放的导出模型，保留既有封存记录。首次setup遗漏scales链接的失败和恢复记录均在本目录归档。
