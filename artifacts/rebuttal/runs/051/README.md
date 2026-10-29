# 051 — Qwen空闲GPU3/4正式速度复测

**已按用户新安排停止本轮派发，转[052持续空闲监测](../052/README.md)。六个完成进程与中断第七进程保留，不并入新安静对照。**

用户确认GPU3/4空闲后启动。GPU3为预先确定的主卡，GPU4为独立复核卡；两卡各完整三轮，各自按BF16归一化。正式工作串行，避免本任务在另一张卡上的模型加载、编译或恢复拷贝干扰计时。

[实时状态](results/worker.json) · [冻结设计](configs/design.json) · [正式结果合表](../../responses/reviewer2/qwen_quiet_speed_run051/README.md) · [原协议完整精度](../../responses/reviewer2/qwen_original_quality_run050/README.md)

四组核心配置：B8/S2048/O64、B4/S2048/O64、B8/S4096/O64、B8/S2048/O128，共三种独立Prefill-only与四种P+D。BF16、HAWQ/DASO各均衡及速度点；PP共同速度端点只测一次。所有工作点、输入token与049相同，050原协议任务精度和049逐配置NLL直接沿用，不重新选点。

每卡三轮、每轮BF16首尾；轮换策略及负载顺序。每case五次预热、十次正式计时，与049同Graph B4/B8/B16、上下文4232、token预算32768、KV6GiB。共54正式进程、234个case轮次、2340次正式计时，另存1170次预热。初始化、内核编译、Graph捕获与权重审计都在计时之外。

按卡分别报告TTFT、TPOT、E2E、包含恢复的cycle，以及三轮范围和5%波动标记；不把不同卡的耗时混合为一个结果，不按是否领先HAWQ决定保留轮次。正式卡上若发现外部计算进程，保留该尝试并待空闲后同参数重试；不结束他人的任务。

权重以硬链接复用050八个不可变checkpoint，无新增权重物理复制；Triton缓存副本只复制路径索引、硬链接已有二进制，新增缓存写本目录。所有命令、代码快照、设备遥测、逐次计时、逐case完成输出和失败尝试保留。旧主表、旧测速与暂估表不改写。

[开发计划180](../../../../dev/plans/180_qwen_quiet_speed_retest_plan.md) · [开发记录180](../../../../dev/impls/180_qwen_quiet_speed_retest_impl.md)
