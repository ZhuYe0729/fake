# Run014: W4A16 TPOT diagnosis

结论：W4A16 的 GPU 计算收益存在。原 eager 完整执行受到 CPU 发射、临时张量操作等开销限制；旧记录还显示 CPU 频率与 TPOT 共同波动。原 run013 的 TPOT 不能代表优化部署后的量化性能。

## 固定真实 KV 的控制

GPU0 / NUMA0 CPU0–7，B8；从真实生成末步冻结输入、positions=2110、seq_lens=2111。包括模型全部层和 lm_head，排除调度/采样。每项5组×30次，计时无 profiler。

| 方法 | 普通前向 ms | Graph 前向 ms | Graph 最大 logits 绝对差 |
| --- | ---: | ---: | ---: |
| BF16 | 15.456 | 12.611 | 0.0 |
| W4A16 NVFP4 | 10.106 | 6.376 | 0.0 |
| W4A16 INT4 Marlin | 16.930 | 6.277 | 0.0 |

单次 eager/direct-dispatch 测量受顺序与 CPU 频率影响；direct-dispatch 位于 profiler 之后，不能单凭该对照估计阶段管理的因果耗时。下面同进程交替实验是更强的 Graph 干预。

## 正常生成，同进程交替开关 Graph

原生 vLLM FULL_DECODE_ONLY，compilation level=0；B8/S2048/O64、6GiB BF16 KV，与 run013 相同输入。均匀策略关闭无用的阶段切换。每个模型只初始化一次，两模式各3次 warmup，随后4组 ABBA/BAAB，共每模式8次正式生成。只切换 decode wrapper 的 replay / runnable，保持权重、引擎和 metadata 构造一致。

| 方法 | Eager TPOT ms | Graph TPOT ms | Graph 下相对 BF16 |
| --- | ---: | ---: | ---: |
| BF16 | 16.867 | 13.517 | 1.000× |
| W4A16 NVFP4 | 15.304 | 7.301 | 1.851× |
| W4A16 INT4 Marlin | 18.124 | 7.147 | 1.891× |

另有两轮倒序的独立进程控制：三模型 × eager / frozen-eager / native-graph，18个进程，每进程3 warmup+8 timed。全部保留在 generation_controls.csv；探路 Graph 不计入两轮汇总。

## 运行时证据与限制

- 固定前向每次包含688个 GPU kernel、144个被量化 Linear；量化缩短了 Linear kernel，完整 eager 仍逐次经过 Python/C++ 调用，不能把权重压缩倍率等同于 TPOT 加速倍率。
- 独立 profiler 的 aten::empty 调用/前向：BF16=361；W4A16 NVFP4=505；W4A16 INT4 Marlin=1657。这是张量创建算子次数，包含零长度占位张量，不表示相同次数的 cudaMalloc。INT4 通用绑定存在较多这类 CPU 操作；未将 profiler 下的耗时直接代入正式 TPOT。
- 原 run013 BF16 round2 首尾：CPU约2.1/2.7GHz时 TPOT约18.04/13.97ms；GPU时钟相近。频率只在计时外采样，属于关联证据；没有单独锁 CPU 时钟证明精确因果份额。
- 均匀 W4A16 的 host 权重缓存为零、前后阶段共享 GPU 权重；结果不支持逐 token H2D 搬权重的解释。
- 三模型固定输入 Graph logits 与 eager 最大绝对差均为零；完整生成 token 变化按原始记录保留，尤其不以 INT4 已知的生成非确定性筛除不利样本。
- Graph 支持在实验局部实现，未修改共享 vLLM/CUDA kernel、量化权重或原表数值。均匀策略可直接使用原生 Graph；DASO/HAWQ 的阶段切换和权重地址生命周期仍需一致处理，不能把本轮 Graph 基线与旧 eager 异构策略拼成新的比较主表。

## 文件与复现

- `code/diagnose_decode.py`：真实固定 KV、单独 trace、静态回放。
- `code/generation_control.py` / `control_queue.py`：正常生成的两轮控制。
- `code/paired_generation.py`：同进程 ABBA 干预；`code/launch.py` 保存精确命令、代码快照及退出码。
- `results/fixed_decode.csv`、`generation_controls.csv`、`paired_generation.csv`、`operator_counts.csv`、`historical_cpu_telemetry.csv`：从原始 JSON/trace 生成。
- `results/audit.json`：检查样本数、固定 KV、Graph 误差及原 run013 manifest 未改变。
- 原理参考：[vLLM CUDA Graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs/)、[Marlin 作者实现](https://github.com/IST-DASLab/marlin)。实际行为以本机固定代码与测量为准。
