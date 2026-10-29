# Qwen3-8B：统一 Graph 运行时后的 Table 2

17行187项指标完整。run015 在新画像上重新分配并冻结选点；相同有效map继承完整质量，新map补齐完整任务。旧表和固定旧配置的重测另行保留。

同一RTX 5090 GPU0、NUMA0 CPU0–7，B8/S2048，Prefill O1、P+D O64，6GiB BF16 KV，无prefix/chunked。所有方法使用同一按层复用权重缓冲区、异步H2D与原生 FULL_DECODE_ONLY Graph；prefill仍为eager。三轮独立进程，每轮3 warmup+8 timed、BF16首尾锚点，表中为各轮归一化加速比的中位数。本任务不并行其他GPU质量作业；其他用户作业仅在计时外采样记录，不按这些记录筛除样本。

ARC-C为acc_norm，ARC-E/Wino/MMLU为acc；全部zero-shot完整测试。CNN/DM1000、DialogSum1500、IWSLT333，最多256个新token、允许EOS。准确率/ROUGE-L为百分数，IWSLT为SacreBLEU。

| Model | Method | ARC-C | ARC-E | Wino. | MMLU | Prefill TTFT x | CNN/DM | DialogSum | IWSLT | P+D TTFT x | P+D TPOT x | P+D E2E x |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | BF16 | 56.91 | 83.46 | 68.03 | 72.87 | 1.00 | 22.81 | 26.96 | 40.13 | 1.00 | 1.00 | 1.00 |
| Qwen3-8B | Dense NVFP4 | 54.61 | 82.37 | 67.96 | 70.97 | 1.70 | 22.35 | 27.08 | 40.20 | 1.70 | 1.10 | 1.38 |
| Qwen3-8B | W4A16 NVFP4 | 56.48 | 83.54 | 69.22 | 71.92 | 0.99 | 22.55 | 27.44 | 40.12 | 0.99 | 1.83 | 1.23 |
| Qwen3-8B | W4A16 INT4 Marlin | 52.65 | 80.09 | 65.35 | 71.16 | 1.00 | 21.96 | 26.35 | 40.02 | 1.00 | 1.82 | 1.23 |
| Qwen3-8B | Sparse BF16 | 44.45 | 73.91 | 64.25 | 54.10 | 1.53 | 21.45 | 21.29 | 32.30 | 1.54 | 0.77 | 1.08 |
| Qwen3-8B | Sparse NVFP4 | 28.16 | 53.87 | 55.41 | 30.80 | 1.86 | 13.53 | 5.32 | 0.91 | 1.85 | 1.18 | 1.49 |
| Qwen3-8B | INT8 W8A8 | 57.08 | 82.28 | 67.72 | 72.55 | 1.89 | 22.79 | 27.27 | 39.79 | 1.90 | 1.39 | 1.65 |
| Qwen3-8B | SmoothQuant INT8 | 57.17 | 83.54 | 68.98 | 72.61 | 1.89 | 22.70 | 27.18 | 39.32 | 1.90 | 1.39 | 1.64 |
| Qwen3-8B | FP8 W8A8 | 55.72 | 83.16 | 67.96 | 72.85 | 1.69 | 22.79 | 26.97 | 40.07 | 1.69 | 0.74 | 1.09 |
| Qwen3-8B | SmoothQuant FP8 | 57.00 | 83.71 | 68.43 | 72.83 | 1.69 | 22.83 | 27.03 | 40.01 | 1.70 | 0.75 | 1.10 |
| Qwen3-8B | HAWQ-v3 INT4, latency target=.80 BF16 (PTQ) | 56.91 | 83.08 | 67.88 | 72.60 | 1.32 | 22.74 | 27.27 | 40.27 | 1.38 | 1.21 | 1.30 |
| Qwen3-8B | HAWQ-v3 FP4, latency target=.80 BF16 (PTQ) | 56.83 | 83.42 | 68.82 | 72.83 | 1.32 | 22.83 | 27.35 | 40.64 | 1.40 | 1.19 | 1.30 |
| Qwen3-8B | HAWQ FP4 minimum-cost control, q=.07 | 55.89 | 81.52 | 69.61 | 71.67 | 2.19 | 22.66 | 27.11 | 39.49 | 2.20 | 1.57 | 1.87 |
| Qwen3-8B | DASO, 5 candidates, fresh-profile DP | 53.58 | 80.89 | 66.69 | 69.04 | 1.68 | 21.36 | 24.58 | 37.39 | 1.95 | 1.85 | 1.90 |
| Qwen3-8B | DASO + 8bit, fresh-profile DP | 57.76 | 82.66 | 68.59 | 71.77 | 1.98 | 22.62 | 26.71 | 39.63 | 1.79 | 1.40 | 1.59 |
| Qwen3-8B | DASO + 8bit + exact/validation, epsilon=.01 | 57.17 | 83.54 | 68.98 | 72.61 | 1.89 | 22.45 | 26.87 | 39.58 | 1.69 | 1.38 | 1.55 |
| Qwen3-8B | DASO + 8bit + exact/validation, epsilon=.04 | 56.06 | 81.02 | 69.22 | 70.34 | 2.21 | 21.67 | 25.44 | 38.58 | 1.97 | 1.57 | 1.78 |

## 三轮速度范围

以下是重复测量最小值–最大值，不是统计置信区间。

| Method | Prefill TTFT x | P+D TTFT x | P+D TPOT x | P+D E2E x |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 1.00–1.00 | 1.00–1.00 | 1.00–1.00 | 1.00–1.00 |
| Dense NVFP4 | 1.70–1.70 | 1.70–1.71 | 1.10–1.16 | 1.38–1.42 |
| W4A16 NVFP4 | 0.99–0.99 | 0.99–0.99 | 1.80–1.84 | 1.22–1.23 |
| W4A16 INT4 Marlin | 0.99–1.00 | 0.99–1.00 | 1.81–1.86 | 1.23–1.25 |
| Sparse BF16 | 1.53–1.53 | 1.53–1.54 | 0.77–0.80 | 1.08–1.09 |
| Sparse NVFP4 | 1.85–1.86 | 1.85–1.85 | 1.17–1.20 | 1.48–1.50 |
| INT8 W8A8 | 1.88–1.90 | 1.89–1.91 | 1.36–1.45 | 1.62–1.68 |
| SmoothQuant INT8 | 1.89–1.89 | 1.90–1.90 | 1.36–1.39 | 1.62–1.65 |
| FP8 W8A8 | 1.69–1.69 | 1.69–1.70 | 0.74–0.76 | 1.09–1.10 |
| SmoothQuant FP8 | 1.69–1.69 | 1.70–1.70 | 0.74–0.76 | 1.10–1.10 |
| HAWQ-v3 INT4, latency target=.80 BF16 (PTQ) | 1.32–1.32 | 1.38–1.38 | 1.21–1.24 | 1.30–1.31 |
| HAWQ-v3 FP4, latency target=.80 BF16 (PTQ) | 1.32–1.32 | 1.40–1.40 | 1.18–1.21 | 1.30–1.31 |
| HAWQ FP4 minimum-cost control, q=.07 | 2.19–2.19 | 2.19–2.20 | 1.56–1.61 | 1.87–1.90 |
| DASO, 5 candidates, fresh-profile DP | 1.67–1.68 | 1.95–1.95 | 1.81–1.88 | 1.89–1.92 |
| DASO + 8bit, fresh-profile DP | 1.98–1.98 | 1.78–1.79 | 1.39–1.43 | 1.59–1.61 |
| DASO + 8bit + exact/validation, epsilon=.01 | 1.89–1.89 | 1.69–1.69 | 1.38–1.41 | 1.54–1.56 |
| DASO + 8bit + exact/validation, epsilon=.04 | 2.21–2.21 | 1.97–1.97 | 1.56–1.58 | 1.77–1.78 |

## 解释与边界

- 同一行的prefill和P+D使用各自固定map，与论文原表口径一致；精确policy-ID与质量来源见provenance.json。
- Prefill TTFT为O1完整调用；P+D在FINAL_ONLY过滤前观测实际首token，TPOT=(同次E2E−TTFT)/63。权重恢复单列restore_ms及完整cycle_ms；表中E2E延续原协议，不含请求前恢复。
- Graph依赖稳定地址。按层缓冲区取两阶段表示所需容量的较大者，保留异步搬运；实际显存、host缓存、初始化/捕获开销均另报，不能当作免费双驻留。
- FP4绑定在GPU上读取原动态FP32缩放系数，使用当前stream及Graph安全workspace；量化与GEMM类型不改，24组数值对照为0误差，真实异构map多次生成与搬运权重检查通过。
- 五候选DASO使用Prefill epsilon=.02/P+D epsilon=.07；七候选原DP使用Prefill epsilon=.02/P+D epsilon=.01，均按新画像重新求解。原DP网格仍是保守离散代理，exact仅指加和代理上的精确求解。
- exact/validation是额外流程：候选池包括训练代理/硬件画像确定的阶段统一anchors；按独立validation mean ΔNLL+2SE+.005筛选，再取校准E2E最快点。epsilon单位为nats/token，不是准确率百分点；不承诺跨任务或分布外质量保证。validation/test为不同split；test沿用历史冻结窗口以便比较，其中部分旧map标签已在此前实验中公开，不能称为跨所有轮次从未查看过的新测试集。
- HAWQ为Hessian＋ILP的Qwen3 PTQ移植，非原CNN integer-only/QAT/TVM全栈。INT4低位为Marlin W4A16；FP4低位在prefill用W4A4、decode用W4A16。HAWQ的q是配对速度上限标签，不是其NLL保证。原q配对上限不可行及最小成本放宽逐点保留在reanalysis_pool.json；主表HAWQ采用预定.80×BF16校准E2E目标，经BF16加和偏移和5%余量换算为线性上限。另报.95/.80/.65全部点；目标不构成实测E2E保证。
- INT4的已知生成输出变化保留，不按token签名筛选速度。所有方法固定输入和输出长度；该协议为离线同阶段批次，未宣称任意混合在线调度支持。

[CSV](main_table.csv) · [LaTeX](main_table.tex) · [逐项来源](provenance.json) · [固定旧配置的Graph重测](../../../runs/015/results/fixed_main_table.md) · [逐轮速度](../../../runs/015/results/speed_repeats.csv) · [开销](../../../runs/015/results/runtime_overhead.csv) · [HAWQ全部预算](../../../runs/015/results/hawq_latency_controls.csv) · [独立test约束](../../../runs/015/results/selected_test_constraints.csv) · [质量–速度图](../../../runs/015/results/heldout_frontier.pdf) · [旧eager表](../../../runs/013/results/table/README.md)

主表组与补充组分别有各自三轮BF16首尾锚点；固定旧配置对照中的共有map沿用主表组测量，其余map使用补充组测量。
