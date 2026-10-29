# 两Llama补SmoothQuant-FP8控制（R2 Exi5 / R3 LrT2）

2026-10-27，161夜间授权后续；独立run042/043，不改原主表。

1. 沿用017/018已有目标模型128-context activation scales及固定alpha=.5；调用原有sq_fp8导出/执行路径，补全uniform SQ-FP8基线，不调alpha、不重拟合DASO、不将第九候选自动加入原八候选空间。
2. 每模型新增完整四项MC（ARC-C/ARC-E/Wino/MMLU，原batch8、Llama3.1为3GiB KV/Llama2为10GiB KV、zero-shot全样本）、三项完整生成（原PMPD/BOS/greedy256/EOS、原batch8/4）、64窗口Prefill及P+D NLL。所有输入/源码/协议冻结，原BF16/INT8/SQ-INT8/FP8质量按SHA复用。
3. 与四个原参考共五map，B8/S2048/O1和64、原KV6/10GiB、native decode Graph，统一同卡三轮3warmup/5timed、BF16首尾锚点，包含原始输出和精确权重检查。
4. 037基线复核通过后空闲卡并行质量；给035/036/038/039/040/041正式测速让路；自身正式速度等041完成，两模型按run顺序。每任务使用现有测量/GPU/export锁。
5. 独立五行完整表和来源，不混入原封存表。不因新基线强于DASO而删去；仅最后删除自己的可重建权重二进制，保留全部其他产物。

验证：原模型/协议/SHA不变，MC完整样本数1172/2376/1267/14042、batch8，GEN1000/1500/333，NLL64；每map两case/三轮同卡实测，所有结果包括不利结果保留。既有公开benchmark不是新的独立盲测。
