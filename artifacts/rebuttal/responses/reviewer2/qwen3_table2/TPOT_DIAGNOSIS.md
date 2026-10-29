# Qwen3-8B TPOT 运行时诊断

run014 已定位主要瓶颈：原 eager 路径逐 token 的 CPU 调用和 kernel 发射开销掩盖了 W4A16 的 GPU 收益；旧记录还显示 CPU 频率与 TPOT 共同波动。

GPU0、B8/S2048/O64、6GiB BF16 KV、固定输入。每个模型只初始化一个原生 vLLM 引擎，同进程 ABBA/BAAB 交替开关 decode Graph，各模式8次正式生成；表中为中位 TPOT。

| 方法 | Eager ms/token | Graph ms/token | Graph 下相对 BF16 |
| --- | ---: | ---: | ---: |
| BF16 | 16.87 | 13.52 | 1.00× |
| W4A16 NVFP4 | 15.30 | 7.30 | 1.85× |
| W4A16 INT4 Marlin | 18.12 | 7.15 | 1.89× |

三模型固定真实 KV 前向中，Graph 与 eager 的 logits 最大绝对差均为0；BF16/NVFP4 的完整生成 token 在同进程两模式中完全一致。INT4 完整生成仍有已知输出变化，全部保留，未按输出签名筛选速度样本。

另有18个独立进程完成两轮倒序控制，分别测试原 eager、移除均匀策略阶段管理的 eager、原生 Graph。移除阶段管理本身并不足以消除 eager 的慢速和波动。

主表的速度仍是 run013 eager 协议下的实测。本轮不改量化权重或质量分数；DASO/HAWQ 存在阶段切换和权重地址生命周期问题，需要同样处理运行时再比较，不能把 Graph 基线与旧 eager 异构结果混排。

[完整报告](../../../runs/014/REPORT.md) · [逐项审计](../../../runs/014/results/audit.json) · [同进程数据](../../../runs/014/results/paired_generation.csv) · [两轮控制](../../../runs/014/results/generation_controls.csv)


## Run015 后续状态

统一运行时、新画像和重新分配后的完整主表已更新至 [README](README.md)。上文为run014历史诊断；旧eager表保存在run013，不再用于新主表排名。
