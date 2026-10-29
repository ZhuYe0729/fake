## 2026-10-27 — 确定主表追加范围
- 根据用户要求，保留原19行，追加两项已完成P+D探索配置和三项同协议参考；不额外运行GPU任务。
- 新DASO行沿用旧独立Prefill-only SQ-INT8指标，P+D自身MC单列诊断，避免把两场景混成同一配置。预算失败和完整敏感性结果继续公开保留。
- 新建plan159；实施、来源核验和主表完整性结果待完成后追加。

## 2026-10-27 — 24行主表与来源核对完成
- 更新reviewer2/qwen3_table2_run016的README、main_table.csv和provenance：原19行保留，追加032 BF16/INT8/SQ-INT8三参考，以及032的q=.01/wP=1、033的q=.03/wP=32两个P+D探索点，共24行、312项指标。未新增GPU任务。
- 两个新点生成Avg30.01239/29.95118，E2E为各自BF16锚点的1.88238/2.02616倍、同轮INT8的1.15121/1.24027倍；TTFT/TPOT/E2E均从原三轮首尾BF16按指标重算，范围和绝对时间保存在new_speed_rounds.csv及new_point_details.csv。
- 新DASO行Prefill-only沿用旧DASO+8bit exact/validation ε=.01的全SQ-INT8，精度和独立速度逐值复用；P+D自身MC69.56066/69.76498单列诊断，未代替主表Prefill-only的70.57590。q/权重仅描述新P+D选择，不声称重新优化了Prefill-only。
- 两点真实NLL保护值.03101/.03350均不满足对应名义预算；README、CSV、JSON保留False标志，行名使用exploratory而非validation通过。完整8标签含差点/别名仍在独立敏感性表，旧INT8和本轮参考差异也注明来源。
- 新增188余行可重建脚本append_run032_033.py，原README/CSV/provenance字节快照和SHA保存于reference/before_run032_033；较早17行目录被015审计覆盖，因此整个目录不修改。
- 验证通过：原19行CSV字节前缀及Markdown数值行不变；24行312项指标均有限、两类Avg由未舍入值重算；97份来源SHA、各新行三轮速度和完整两阶段144模块分配核对；重复运行的5份交付文件哈希完全一致。015封存交付及032/033审计SHA未变。
