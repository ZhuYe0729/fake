# 排名与约束可靠性复核证据（run012）

[三个问题的直接说明](../REPRODUCIBILITY_FOLLOWUP.md) · [完整实验报告](../runs/012/REPORT.md) · [专项审计](../runs/012/results/final_audit.json)。

| 人工reviewer | 证据包 | 相关内容 |
| --- | --- | --- |
| AK6a / reviewer1 | [run012](reviewer1/reproducibility_run012/README.md) | 实际质量筛选的含义、质量/时延可行性与回退 |
| Exi5 / reviewer2 | [run012](reviewer2/reproducibility_run012/README.md) | 强HAWQ比较的跨进程复核与排名变化 |
| LrT2 / reviewer3 | [run012](reviewer3/reproducibility_run012/README.md) | 时延预测/校准误差、NUMA/CPU/GPU元数据、驻留/allocator控制 |
| gkP9 / reviewer5 | [run012](reviewer5/reproducibility_run012/README.md) | 反向时延约束、冻结后test、经验上界与不可行结果 |

本轮不覆盖AI reviewer。原21份夜间证据包保持原样；新的4份包使用独立前缀，来源manifest和副本SHA记录在各source.json。
