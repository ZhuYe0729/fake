# Qwen3-8B：论文 Table 2 主表

状态：现有17行全部补齐，共187个指标。策略在本轮开始前固定；新增3组完整likelihood任务及2组完整生成任务，其余完整精度结果按相同map继承。

速度全部来自run013：同一RTX 5090 GPU0，CPU0–7/node0，B8/S2048，Prefill O1、P+D O64，6GiB BF16 KV，eager、无prefix/chunked。三轮独立进程完整测量，每轮3 warmup+8 timed，BF16首尾锚点；主表取各轮相对本轮BF16加速比的中位数。
准确率与ROUGE-L为百分数，IWSLT为SacreBLEU。ARC-C用acc_norm，ARC-E用acc；likelihood为完整zero-shot任务，生成任务CNN/DM1000、DialogSum1500、IWSLT333。

| Model | Method | ARC-C | ARC-E | Wino. | MMLU | Prefill TTFT x | CNN/DM | DialogSum | IWSLT | P+D TTFT x | P+D TPOT x | P+D E2E x |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | BF16 | 56.91 | 83.46 | 68.03 | 72.87 | 1.00 | 22.81 | 26.96 | 40.13 | 1.00 | 1.00 | 1.00 |
| Qwen3-8B | Dense NVFP4 | 54.61 | 82.37 | 67.96 | 70.97 | 1.70 | 22.35 | 27.08 | 40.20 | 1.70 | 0.51 | 0.84 |
| Qwen3-8B | W4A16 NVFP4 | 56.48 | 83.54 | 69.22 | 71.92 | 0.99 | 22.55 | 27.44 | 40.12 | 0.99 | 0.85 | 0.92 |
| Qwen3-8B | W4A16 INT4 Marlin | 52.65 | 80.09 | 65.35 | 71.16 | 1.00 | 21.96 | 26.35 | 40.02 | 1.00 | 0.71 | 0.85 |
| Qwen3-8B | Sparse BF16 | 44.45 | 73.91 | 64.25 | 54.10 | 1.53 | 21.45 | 21.29 | 32.30 | 1.53 | 0.74 | 1.05 |
| Qwen3-8B | Sparse NVFP4 | 28.16 | 53.87 | 55.41 | 30.80 | 1.84 | 13.53 | 5.32 | 0.91 | 1.85 | 0.44 | 0.77 |
| Qwen3-8B | INT8 W8A8 | 57.08 | 82.28 | 67.72 | 72.55 | 1.90 | 22.79 | 27.27 | 39.79 | 1.92 | 0.59 | 0.96 |
| Qwen3-8B | SmoothQuant INT8 | 57.17 | 83.54 | 68.98 | 72.61 | 1.89 | 22.70 | 27.18 | 39.32 | 1.91 | 0.42 | 0.75 |
| Qwen3-8B | FP8 W8A8 | 55.72 | 83.16 | 67.96 | 72.85 | 1.69 | 22.79 | 26.97 | 40.07 | 1.70 | 0.59 | 0.90 |
| Qwen3-8B | SmoothQuant FP8 | 57.00 | 83.71 | 68.43 | 72.83 | 1.69 | 22.83 | 27.03 | 40.01 | 1.69 | 0.51 | 0.85 |
| Qwen3-8B | HAWQ-v3 INT4, q=.01 (PTQ adaptation) | 55.72 | 83.08 | 69.22 | 72.67 | 1.72 | 22.80 | 27.23 | 39.73 | 1.62 | 0.51 | 0.83 |
| Qwen3-8B | HAWQ-v3 FP4, q=.01 (PTQ adaptation) | 56.31 | 84.05 | 69.46 | 72.44 | 1.73 | 22.55 | 27.10 | 39.25 | 1.64 | 1.06 | 1.32 |
| Qwen3-8B | HAWQ FP4 minimum-cost control, q=.07 | 55.89 | 81.52 | 69.61 | 71.67 | 2.19 | 22.49 | 27.30 | 40.18 | 1.69 | 1.04 | 1.33 |
| Qwen3-8B | DASO, original 5 candidates (representatives) | 54.44 | 81.40 | 66.69 | 68.96 | 1.67 | 21.67 | 25.44 | 38.58 | 1.97 | 0.80 | 1.20 |
| Qwen3-8B | DASO + 8bit, original DP (representatives) | 56.23 | 81.99 | 67.64 | 71.51 | 1.98 | 22.52 | 26.80 | 39.49 | 1.79 | 0.79 | 1.15 |
| Qwen3-8B | DASO + 8bit + exact/validation, epsilon=.01 | 57.17 | 83.54 | 68.98 | 72.61 | 1.89 | 22.81 | 26.96 | 40.13 | 1.00 | 1.00 | 1.00 |
| Qwen3-8B | DASO + 8bit + exact/validation, epsilon=.04 | 55.03 | 81.99 | 69.93 | 71.10 | 2.20 | 22.60 | 26.72 | 39.54 | 1.79 | 0.91 | 1.23 |

## 速度重复范围

以下为三轮加速比的最小值–最大值，用于展示重复测量变化，不是统计置信区间。

| Method | Prefill TTFT x | P+D TTFT x | P+D TPOT x | P+D E2E x |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 1.00–1.00 | 1.00–1.00 | 1.00–1.00 | 1.00–1.00 |
| Dense NVFP4 | 1.69–1.70 | 1.70–1.70 | 0.50–0.57 | 0.84–0.88 |
| W4A16 NVFP4 | 0.99–0.99 | 0.99–0.99 | 0.85–0.96 | 0.92–0.97 |
| W4A16 INT4 Marlin | 1.00–1.00 | 0.99–1.00 | 0.71–0.81 | 0.85–0.90 |
| Sparse BF16 | 1.52–1.53 | 1.53–1.54 | 0.73–0.83 | 1.03–1.10 |
| Sparse NVFP4 | 1.84–1.85 | 1.85–1.85 | 0.44–0.47 | 0.77–0.77 |
| INT8 W8A8 | 1.88–1.90 | 1.90–1.92 | 0.58–0.67 | 0.96–1.02 |
| SmoothQuant INT8 | 1.88–1.90 | 1.89–1.91 | 0.42–0.52 | 0.74–0.84 |
| FP8 W8A8 | 1.69–1.69 | 1.70–1.70 | 0.52–0.70 | 0.85–1.05 |
| SmoothQuant FP8 | 1.69–1.69 | 1.69–1.70 | 0.51–0.80 | 0.84–1.11 |
| HAWQ-v3 INT4, q=.01 (PTQ adaptation) | 1.72–1.74 | 1.61–1.62 | 0.50–0.58 | 0.82–0.87 |
| HAWQ-v3 FP4, q=.01 (PTQ adaptation) | 1.73–1.73 | 1.64–1.64 | 0.77–1.24 | 1.10–1.42 |
| HAWQ FP4 minimum-cost control, q=.07 | 2.19–2.19 | 1.69–1.69 | 0.97–1.18 | 1.25–1.42 |
| DASO, original 5 candidates (representatives) | 1.67–1.67 | 1.97–1.97 | 0.80–0.89 | 1.20–1.25 |
| DASO + 8bit, original DP (representatives) | 1.98–1.98 | 1.78–1.79 | 0.77–0.90 | 1.13–1.22 |
| DASO + 8bit + exact/validation, epsilon=.01 | 1.88–1.90 | 1.00–1.00 | 1.00–1.00 | 1.00–1.00 |
| DASO + 8bit + exact/validation, epsilon=.04 | 2.20–2.20 | 1.78–1.79 | 0.78–1.05 | 1.14–1.37 |

## 协议与方法说明

- 与原论文Table 2一致，同一行的Prefill-only与P+D使用各自的固定配置；完整policy-ID与质量来源映射见provenance.json及冻结design.json。
- Prefill TTFT采用O1完整调用时间。P+D TTFT在EngineCore输出进入FINAL_ONLY过滤前实际观测；TPOT按同一次 `(E2E−TTFT)/63` 计算，三项均有逐次原始数据。
- 任务精度与固定负载速度分别评测。完整likelihood沿用batch4/3GiB KV（WikiText为batch1）；生成沿用batch8/6GiB KV、greedy最多256个新token且允许EOS。表中O64加速比不表示完整下游任务的运行加速。
- 延续原论文计时范围：请求前的prefill权重恢复不计入E2E，另保留restore_ms与包含恢复的cycle_ms。质量评测先结束，随后执行三轮测速，未用并行质量任务期间的诊断计时填主表。
- 原生INT4 Marlin在同输入重复生成时出现token输出变化，首次严格输出一致性断言中止的尝试及后续诊断均保留。正式测速固定输入与生成长度、忽略EOS，不以输出token逐字相同筛选速度；各次输出签名/后续逐token记录与output_variation.csv公开保留。该差异不解释成质量稳定性保证。
- HAWQ行是Hessian＋ILP分配方法的Qwen3 PTQ移植，非原CNN integer-only/TVM全栈复现。INT4候选是Marlin W4A16；FP4低位动作在Prefill用NVFP4 W4A4，在Decode用FP4权重W4A16。
- INT4候选行不表示模型全部采用INT4；当前该行P+D配置由72个INT8与72个BF16模块组成，没有选中INT4。
- HAWQ的q为构造速度上限时配对的DASO预算标签，不代表HAWQ质量保证。INT4 q=.01的P+D原上限不可行；FP4 q=.07两场景原上限均不可行，均记录原上限并放宽至各自最小可达Linear成本，不能声称达到原目标。
- FP4 q=.07的P+D最终为阶段统一配置，完整map与run001已测对照相同，生成分数继承该点；其Prefill分数本轮实际补测。
- DASO + 8bit为原五候选加入SQ-INT8和FP8，共七候选。原DP行采用Prefill epsilon=.02/P+D epsilon=.01；五候选行采用Prefill epsilon=.02/P+D epsilon=.07的已有代表点，均保持原配置。
- exact/validation行来自预先固定的扩展求解候选池，再按独立validation质量及校准速度筛选。epsilon=.01选择Prefill SQ-INT8/P+D BF16；epsilon=.04选择两个异构配置。该额外步骤独立于原算法，不是新kernel，也不承诺跨任务质量保证。
- 原DP的epsilon约束预测ΔNLL，validation的epsilon用于验证损失加余量的筛选；单位均为nats/token，不能解释成MMLU允许下降的百分点。
- 旧主表、旧单轮速度及所有失败记录保留于run013/reference/table_before及旧runs；新表不按输赢选择某一轮，也未根据新下游分数调整策略。

[主表CSV](main_table.csv) · [LaTeX](main_table.tex) · [逐项来源/哈希](provenance.json) · [逐轮速度CSV](../../../runs/013/results/speed_repeats.csv) · [run013报告](../../../runs/013/REPORT.md)

## TPOT 运行时补充诊断

run014 已确认 W4A16 的 GPU 加速被 eager 执行开销掩盖，并验证均匀基线的原生 CUDA Graph 路径。本表继续保留 run013 的原始协议和数值；Graph 基线与异构策略需采用一致运行时后才能更新排名。见 [诊断结果与适用范围](TPOT_DIAGNOSIS.md)。
