# Run019：固定主配置跨负载迁移

对应R2 Exi5、R3 LrT2、R4 QboE关于负载数和decode长度的concern。依plan151继续；先等两个原Llama核心主表完成，不阻塞其正式测量。

- 六种B/S/O含B8/S512/O256与B1/S512/O256；11行标签/10套唯一固定map，保留旧DASO incumbent、当前DASO-8、HAWQ中心/硬件端点、SQ-INT8/FP8/W4A16/BF16及简单phase对照。
- 每负载64窗口教师强制decode NLL，全部真实新测；三轮GPU0速度，每负载3warmup+5timed，BF16首尾归一化。
- 一个策略进程依次运行六种负载，B1/B8分别捕获native decode Graph；每种负载检查整个batch的真实KV容量。
- 没有按扩展负载结果重新优化或筛选；公开WikiText test语料历史已使用，新的确定性源片段不是未见新语料。
- 失败、部分输出与所有源码版本均保留。结果完成后见REPORT.md；启动后台控制器不等于完成。

[冻结设计](configs/design.json) · [执行代码](code/driver.py)
