## 2026-10-28 - 六负载直接展示强基线质量与速度
- 目的：解决原多负载摘要仅比较INT8速度、不能直接展示与HAWQ质量–速度取舍的问题。
- 新增responses/reviewer2/qwen_workload_comparison：六种P+D负载×九方法的54行完整Markdown/HTML/CSV，含实际ΔNLL±2SE、TTFT、TPOT、E2E和完整恢复cycle；附30行强对照、摘要、PDF/SVG/PNG质量–速度图及来源。主表命名与固定policy签名一致，没有逐负载重选配置。
- 修改results176/build_results.py，总览第3节改为六负载×五方法的30行直接强对照，统一相对BF16；responses索引和英文草稿的中文来源指向新完整表。英文正文不变，三模型主结果CSV不变。旧文档与生成/审计文件保留在results177/reference。
- 精度使用每个实际B/S/O的64窗口逐token teacher forcing，明确不同于三任务生成Avg；后者没有逐负载重跑，未将中心分数复制到新负载。质量和速度同取040，未混拼048更有利的中心速度，也未改动任何原运行结果。
- 保留结果取舍：DASO Balanced的E2E中位数6/6快于HAWQ Max-speed，DASO Max-speed为5/6；HAWQ通常NLL更低；全部九方法中DASO最快4/6，简单阶段组合最快2/6。新负载未承诺原代理标签的实际预算保证，不作全面支配或显著性声明。
- 验证：108份来源SHA、040原84份证据SHA；54行原始paired NLL/SE和三轮速度中位数重算；Markdown/HTML/总览单元格匹配；165本地链接通过；图已视觉检查。总览17表133行，三模型主表原始数值与015–018原manifest保持一致。审计见results177/audit.json，无新增GPU测量。

## 2026-10-28 - 中心两点速度排序不稳定的跟进
- 用户指出中心Balanced比Max-speed更快且平均NLL更小。复核040/048共12组原始进程记录及每组5次计时：040 Balanced三轮E2E为1034.73/988.28/989.96 ms，Max-speed为1072.36/1065.33/966.39 ms；主要变化在Decode，不能把中心两点排序解释为稳定结论。
- 048对应E2E约991/984 ms，但两轮中心输入的16384个token均不同，Graph捕获尺寸及引擎容量也不同。因此048不是严格复测，不能以其更稳定或更有利的数值覆盖040；CPU/频率差异仅为观测线索，未定位波动因果。
- 新增results177/center_speed_followup下的逐轮表与原始来源SHA；在完整Markdown/HTML及总览表旁明确标记中心排序待严格复测，解释Max-speed沿用报告点名、不保证每负载最优，以及±2SE属于NLL质量统计。未更改原数据或按坏点删轮次。
- 更新前文件留存results177/reference/center_speed_followup；两套校验脚本重新通过，三模型原指标不变，仍无新增GPU测量。尚需按完全固定协议、预定重复次数的交错测速，不能声称已完成该新实验。
