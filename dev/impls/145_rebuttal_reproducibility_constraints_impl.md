# 145 开发记录

## 2026-10-26 — 启动独立复核
- 用户要求重测排名，并理解validation/时延违约的机制与解决方法。
- 读取001–011封存报告与实际选择代码：质量guard为mean+2SE+.005；时延guard原为实测median×1.05。14%示例来自同配置两批实测，不能直接归因于质量/时延预测误差。
- 机器有2个NUMA节点，GPU0邻接node0，此前未固定CPU/内存；子进程numactl cpuset/membind测试成功。新增145计划与run012验证，不修改既有封存。

## 2026-10-26 — 固定协议与监控检查
- run012冻结六个旧map、3+8计时、四个本地/四个remote进程控制；正式模型计时前配对seed：校准两轮同原011 seed144983，test两轮同新seed144995；原设计副本和修订时间保留。
- NVML首次动态加载缺少sys.modules登记，copy进程启动失败；补注册后ABBA四个copy进程完成，无计时数据被覆盖。node0/node1的512MiB带宽均约50GiB/s，初步不支持单纯NUMA拷贝带宽是主因。
- 首批模型E2E数据有效；监控检查发现LLM.generate会强制FINAL_ONLY，原step末端观察的是最终输出。只将首token观测点移到EngineCore原始输出处理前，新增observer版本标记及断言；未标记版本的早期TTFT字段不进入分解，原E2E/restore及源码快照保留。后续rank/SLA仍按完整E2E评价，不使用该诊断字段选点。

## 2026-10-26 — 完成四轮重测与约束对照
- 28个主模型进程+4个双驻留控制、512次timed生成，4个成功copy进程/160次timed拷贝；40项同输入token检查均一致。DASO5对HAWQ phase-uniform四个本地campaign为3快1慢，同seed下prefill约577/576ms稳定、P+D约1587/1410ms仍变化，不能声称稳定最优。
- node0/node1大块pinned copy相近，完整模型放置差异方向不一致；NUMA未被证实为单一根因。记录CPU/GPU时钟、CPU亲和/实际NUMA页、restore、首token和可用allocator统计，未更改全局频率策略。更多执行波动的因果贡献未定位。
- 原5%规则校准R65 median1751ms、max1972ms，选择阈值1838ms低于新SLA1909ms；新test中位2068ms、11/16次超时。校准最大值×1.02排除该候选，换选exact_q0.02，新test中位1615ms、max1636ms、0/16超时；选择在任何round2计时前冻结。两条规则对80%/65%目标均在预置小候选池内不可行。
- 换选后原validation质量guard约.01135→.05548，不能据此承诺同时满足严格质量预算。没有新质量标签或原质量/时延代理重新拟合；经验结果不提供未来硬保证。双驻留两次波动较小但更慢，记录额外显存/host缓存，不冒充免费修复。
- run012专项审计通过：42个命令、11源码版本，编译vLLM算子及原后端代码保持原SHA；原001–011 manifest SHA链未改。封存385文件，manifest 288bd7d711ad614c70e2fadca12af72d58c57c8ec4477db4a6492aa523dc3ff9；新增4份reviewer证据包核验72副本/24来源链接。
- 新说明artifacts/rebuttal/REPRODUCIBILITY_FOLLOWUP.md与responses/REPRODUCIBILITY_INDEX.md独立于旧夜间汇总；原21包不改。代码/日志/CSV/失败及监控排除记录保留，5份可复现临时导出仍在/dev/shm并有路径/size/mtime记录，方便后续复核。

## 2026-10-26 — 按论文Table 2汇总Qwen主表现状
- 用户明确所需为下游精度与TTFT/TPOT/E2E并列的Qwen主表；原003属于负载覆盖表，不能替代该主表。
- 新增responses/reviewer2/qwen3_table2下的生成脚本、Markdown/CSV/LaTeX和逐项来源；17行沿用已有代表策略，包含8bit基线、强化HAWQ、原五候选/扩展候选DASO及exact/validation版本。只读取001/002/010的完整任务原始JSON，统一引用011最后一轮速度，不以旧HAWQ任务分数填新策略。
- 145份来源文件SHA与质量/速度完整method-map连接核验通过；13列共124个已填指标、63个待补指标。缺项包含强化HAWQ完整任务、最新统一计时的P+D TTFT/TPOT及未纳入011的旧基线速度，明确标注尚非完整终稿。
- ARC-E按原Table 2的acc从原始JSON读取，避免与010报告的acc_norm混用。复核CSV数值有限、17行和LaTeX每行13列；未启动GPU实验，未改动封存runs或旧证据包。
