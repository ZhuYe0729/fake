# Rebuttal补充实验总清单：Llama + Qwen

更新：2026-10-27 05:34 UTC。只覆盖五位人工reviewer；不含AI Reviewer。本文是待补实验与拟交付表格的总览，不表示所有表已经测完，也不直接启动新GPU任务。

**老师的建议应当落实：Qwen回答跨架构泛化，原两个Llama仍需补HAWQ-style、SmoothQuant INT8和FP8等强基线。** 最终应有三个模型各自完整的主表，而不是只在Qwen上比较新方法。

优先级：**P0＝主结论必补；P1＝针对明确concern的补强；P2＝可选控制/条件扩展。** 状态：`已有`、`运行中`、`待补`、`待汇总`、`条件项`。`待`不是0或失败；`—`表示不适用。

当前快照：Qwen015/016已完成并封存，完整19行汇总已交付。Llama run018：核心完成；Llama run017：核心完成。019六负载已完成；020剪枝已完成；021五预算已完成；022预算测速已完成；025接口已完成；026validation样本量已完成；详细状态见[NIGHT151报告](NIGHT151_REPORT.md)及[监督快照](night151/status.json)。

## 表0 — Reviewer AK6a / Exi5 / LrT2 / QboE / gkP9：补实验总览

| 人工reviewer | 主要concern | 对应下方表 | 本轮已补与剩余边界 | 优先级 |
| --- | --- | --- | --- | --- |
| R1 / AK6a | 实测质量预算、异构基线、双表示内存、跨家族 | 1–6、10–11 | 已补三模型主表/预算/内存；NLL预算不保证全部下游指标 | P0 |
| R2 / Exi5 | HAWQ/HAQ、SQ/FP8、单一B/S/O、continuous/chunked batching | 1–4、7–8 | 已补HAWQ/8bit/六负载；生产continuous仍有缺口 | P0；在线扩展为条件项 |
| R3 / LrT2 | 混合精度、Qwen、预测误差、非GEMM、每设备准备成本 | 1–4、9–10、12、15–16 | 已补Qwen/混合精度/误差/成本；LUT与Hybrid泛化区别保留 | P0/P1 |
| R4 / QboE | 30B、搜索时间、简单剪枝、Figure2 | 10、13–14、17 | 已补搜索成本/Wanda/Magnitude/Figure2；完整30B仍缺 | P1；30B需资源 |
| R5 / gkP9 | NAS、跨模块转换、更多动作、反向SLA、相同输入 | 4–6、10–12、16–17 | 已补8候选/反约束/接口；完整NAS移植未开展 | P0/P1 |

来源：[审稿原文](reference/reviewers.md)、[提交PDF](reference/wu.pdf)、[历史证据矩阵](responses/HUMAN_REVIEWER_EVIDENCE.md)。以下编号是本清单编号，论文主表仍指提交PDF中的Table 2。

## 三张主表的统一口径

- 主负载：RTX5090、B8/S2048，Prefill O1、P+D O64；同一模型的所有新速度统一到同一卡、NUMA、KV、Graph/异步搬运协议，三轮独立进程与BF16首尾锚点。另存绝对毫秒、重复范围、恢复时间及完整cycle。
- 主表质量：ARC-C acc_norm、ARC-E/Wino/MMLU acc；CNN/DM与DialogSum为ROUGE-L%，IWSLT为SacreBLEU。准确率、ROUGE与BLEU不混成一个质量保证。各模型模板、tokenizer、数据SHA、样本量固定；完整任务批大小与测速批大小分别记录。
- **原文旧协议表已归档；当前两个Llama主表采用统一新路径重新评测。补充计划表里仍标`原文†`的数值仅供参考。** 旧质量继承须核验等价，本轮核心完整MC/生成优先重跑；新旧结果不能直接合并宣称排名。
- W4A16 NVFP4与原生INT4 Marlin分开命名。HAWQ一律标`Hessian/ILP PTQ适配`；不能称为原CNN QAT/TVM全栈复现。硬件最小成本端点另列，不作为Hessian代理更优的证据。
- `DASO-8`指原五动作 + SQ-INT8 + FP8 + plain INT8。历史某些文件名含daso8但实际只有七动作，不能按名称判定。各模型补自己的局部误差、scales、训练标签/Hessian；不复用Qwen的拟合系数。
- 最终核心比较包括原五候选DASO、新DASO-8、HAWQ-FP4、SQ-INT8、FP8及原uniform点。SQ-FP8、原生INT4/HAWQ-INT4是已有Qwen控制，Llama优先级较低；七候选中间版本主要作为消融，不要求三个模型都重跑其全部历史版本。

## 表1 — Reviewer AK6a / Exi5 / LrT2 / gkP9：Llama-2-7B-Chat完整主表（P0）

本表取自run018统一新路径的15行核心主表，旧论文数值保留在归档，不混入以下新评测。当前核心完成；原生INT4与SQ-FP8等P2控制尚未开展。

[完整来源、协议和逐项CSV](runs/018/REPORT.md)。

| 方法 | ARC-C | ARC-E | Wino. | MMLU | Prefill TTFT × | CNN/DM | DialogSum | IWSLT | P+D TTFT × | TPOT × | E2E × | 状态/优先级 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BF16 | 43.26 | 72.81 | 68.27 | 46.45 | 1.00 | 23.60 | 21.61 | 19.44 | 1.00 | 1.00 | 1.00 | 核心完成 |
| Dense NVFP4 | 42.92 | 71.51 | 67.56 | 44.98 | 1.80 | 23.57 | 21.48 | 17.23 | 1.80 | 1.11 | 1.38 | 核心完成 |
| W4A16 NVFP4 Marlin | 42.92 | 71.72 | 68.19 | 45.80 | 0.99 | 23.73 | 21.77 | 18.85 | 0.99 | 1.58 | 1.21 | 核心完成 |
| Sparse BF16 (SparseGPT) | 34.98 | 63.80 | 63.85 | 28.25 | 1.60 | 15.35 | 13.46 | 3.95 | 1.61 | 0.86 | 1.13 | 核心完成 |
| Sparse NVFP4 (SparseGPT) | 23.55 | 37.16 | 51.62 | 23.25 | 2.00 | 9.66 | 11.11 | 0.73 | 2.00 | 1.20 | 1.51 | 核心完成 |
| INT8 W8A8 | 43.52 | 72.52 | 68.67 | 45.73 | 1.85 | 23.66 | 21.75 | 16.87 | 1.87 | 1.29 | 1.53 | 核心完成 |
| SmoothQuant INT8 | 43.43 | 73.11 | 68.90 | 46.20 | 1.86 | 23.56 | 21.32 | 18.20 | 1.88 | 1.29 | 1.54 | 核心完成 |
| FP8 W8A8 | 43.17 | 72.39 | 67.40 | 46.37 | 1.70 | 23.61 | 21.80 | 19.05 | 1.71 | 0.83 | 1.13 | 核心完成 |
| Uniform SQ-INT8 prefill / W4A16 decode | 43.43 | 73.11 | 68.90 | 46.20 | 1.86 | 23.90 | 21.66 | 18.47 | 1.85 | 1.42 | 1.62 | 核心完成 |
| DASO-5 fixed original balanced | 42.58 | 71.76 | 67.72 | 44.18 | 1.85 | 23.80 | 21.39 | 21.12 | 1.37 | 1.20 | 1.28 | 核心完成 |
| DASO-5 fixed original max_speed | 28.24 | 53.32 | 54.93 | 23.66 | 2.25 | 18.46 | 13.57 | 2.61 | 2.24 | 1.51 | 1.81 | 核心完成 |
| HAWQ-FP4 PTQ r=.80 | 42.49 | 71.89 | 68.03 | 45.88 | 1.33 | 23.84 | 21.60 | 18.12 | 1.43 | 1.22 | 1.32 | 核心完成 |
| Common hardware-minimum FP4/INT8 control | 43.43 | 71.55 | 66.85 | 44.96 | 2.19 | 23.67 | 21.57 | 17.40 | 2.19 | 1.50 | 1.79 | 核心完成 |
| DASO-8 measured-LUT + exact/validation epsilon=0.01 | 43.17 | 72.39 | 67.40 | 46.37 | 1.70 | 23.60 | 21.61 | 19.44 | 1.00 | 1.00 | 1.00 | 核心完成 |
| DASO-8 measured-LUT + exact/validation epsilon=0.04 | 43.43 | 71.55 | 66.85 | 44.96 | 2.19 | 23.66 | 21.75 | 16.87 | 1.87 | 1.29 | 1.53 | 核心完成 |

## 表2 — Reviewer AK6a / Exi5 / LrT2 / gkP9：Llama-3.1-8B-Instruct完整主表（P0）

本表取自run017统一新路径的15行核心主表，旧论文数值保留在归档，不混入以下新评测。当前核心完成；原生INT4与SQ-FP8等P2控制尚未开展。

[完整来源、协议和逐项CSV](runs/017/REPORT.md)。

| 方法 | ARC-C | ARC-E | Wino. | MMLU | Prefill TTFT × | CNN/DM | DialogSum | IWSLT | P+D TTFT × | TPOT × | E2E × | 状态/优先级 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BF16 | 55.72 | 82.24 | 73.88 | 68.44 | 1.00 | 19.34 | 13.48 | 10.70 | 1.00 | 1.00 | 1.00 | 核心完成 |
| Dense NVFP4 | 51.19 | 78.96 | 71.98 | 63.96 | 1.78 | 16.57 | 9.11 | 10.61 | 1.78 | 1.25 | 1.50 | 核心完成 |
| W4A16 NVFP4 Marlin | 53.67 | 80.98 | 73.09 | 66.19 | 1.00 | 17.44 | 9.93 | 10.53 | 1.00 | 2.01 | 1.28 | 核心完成 |
| Sparse BF16 (SparseGPT) | 38.48 | 68.27 | 66.22 | 42.92 | 1.52 | 6.69 | 5.10 | 3.04 | 1.53 | 0.87 | 1.15 | 核心完成 |
| Sparse NVFP4 (SparseGPT) | 26.62 | 46.46 | 54.70 | 23.73 | 1.98 | 4.27 | 0.47 | 0.02 | 1.97 | 1.38 | 1.66 | 核心完成 |
| INT8 W8A8 | 55.55 | 81.57 | 73.95 | 67.66 | 1.89 | 19.58 | 14.01 | 10.89 | 1.92 | 1.51 | 1.72 | 核心完成 |
| SmoothQuant INT8 | 55.46 | 81.86 | 73.88 | 67.67 | 1.90 | 19.50 | 12.94 | 10.71 | 1.92 | 1.54 | 1.73 | 核心完成 |
| FP8 W8A8 | 54.78 | 82.32 | 74.11 | 67.74 | 1.72 | 19.72 | 14.64 | 10.68 | 1.73 | 0.81 | 1.16 | 核心完成 |
| Uniform SQ-INT8 prefill / W4A16 decode | 55.46 | 81.86 | 73.88 | 67.67 | 1.90 | 19.84 | 13.46 | 10.52 | 1.92 | 1.81 | 1.87 | 核心完成 |
| DASO-5 fixed original balanced | 52.05 | 80.47 | 72.38 | 66.27 | 1.61 | 18.65 | 11.85 | 10.64 | 1.65 | 1.41 | 1.53 | 核心完成 |
| DASO-5 fixed original max_speed | 32.94 | 61.45 | 60.85 | 31.23 | 2.24 | 14.58 | 4.29 | 1.36 | 2.22 | 1.59 | 1.89 | 核心完成 |
| HAWQ-FP4 PTQ r=.80 | 55.29 | 82.24 | 73.95 | 68.13 | 1.31 | 18.94 | 13.65 | 10.60 | 1.37 | 1.28 | 1.33 | 核心完成 |
| Common hardware-minimum FP4/INT8 control | 54.44 | 81.10 | 73.80 | 66.48 | 2.28 | 18.83 | 12.26 | 10.54 | 2.29 | 1.73 | 2.01 | 核心完成 |
| DASO-8 measured-LUT + exact/validation epsilon=0.01 | 55.72 | 82.24 | 73.88 | 68.44 | 1.00 | 19.64 | 13.02 | 10.85 | 1.56 | 1.10 | 1.32 | 核心完成 |
| DASO-8 measured-LUT + exact/validation epsilon=0.04 | 55.46 | 81.86 | 73.88 | 67.67 | 1.90 | 18.83 | 12.26 | 10.54 | 2.29 | 1.73 | 2.01 | 核心完成 |

## 表3 — Reviewer AK6a / Exi5 / LrT2 / QboE / gkP9：Qwen3-8B完整主表（P0）

以下19行是015原17行加016新DASO-8的跨campaign完整汇总，各campaign使用自身BF16归一化。不是19行同轮重测；旧质量来源与新完整任务见逐项provenance。

[完整19行主表与来源](responses/reviewer2/qwen3_table2_run016/README.md)。

| 方法 | ARC-C | ARC-E | Wino. | MMLU | Prefill TTFT × | CNN/DM | DialogSum | IWSLT | P+D TTFT × | TPOT × | E2E × | 状态/优先级 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BF16 | 56.91 | 83.46 | 68.03 | 72.87 | 1.00 | 22.81 | 26.96 | 40.13 | 1.00 | 1.00 | 1.00 | 已完成；015/016跨campaign |
| Dense NVFP4 | 54.61 | 82.37 | 67.96 | 70.97 | 1.70 | 22.35 | 27.08 | 40.20 | 1.70 | 1.10 | 1.38 | 已完成；015/016跨campaign |
| W4A16 NVFP4 | 56.48 | 83.54 | 69.22 | 71.92 | 0.99 | 22.55 | 27.44 | 40.12 | 0.99 | 1.83 | 1.23 | 已完成；015/016跨campaign |
| W4A16 INT4 Marlin | 52.65 | 80.09 | 65.35 | 71.16 | 1.00 | 21.96 | 26.35 | 40.02 | 1.00 | 1.82 | 1.23 | 已完成；015/016跨campaign |
| Sparse BF16 | 44.45 | 73.91 | 64.25 | 54.10 | 1.53 | 21.45 | 21.29 | 32.30 | 1.54 | 0.77 | 1.08 | 已完成；015/016跨campaign |
| Sparse NVFP4 | 28.16 | 53.87 | 55.41 | 30.80 | 1.86 | 13.53 | 5.32 | 0.91 | 1.85 | 1.18 | 1.49 | 已完成；015/016跨campaign |
| INT8 W8A8 | 57.08 | 82.28 | 67.72 | 72.55 | 1.89 | 22.79 | 27.27 | 39.79 | 1.90 | 1.39 | 1.65 | 已完成；015/016跨campaign |
| SmoothQuant INT8 | 57.17 | 83.54 | 68.98 | 72.61 | 1.89 | 22.70 | 27.18 | 39.32 | 1.90 | 1.39 | 1.64 | 已完成；015/016跨campaign |
| FP8 W8A8 | 55.72 | 83.16 | 67.96 | 72.85 | 1.69 | 22.79 | 26.97 | 40.07 | 1.69 | 0.74 | 1.09 | 已完成；015/016跨campaign |
| SmoothQuant FP8 | 57.00 | 83.71 | 68.43 | 72.83 | 1.69 | 22.83 | 27.03 | 40.01 | 1.70 | 0.75 | 1.10 | 已完成；015/016跨campaign |
| HAWQ-v3 INT4, latency target=.80 BF16 (PTQ) | 56.91 | 83.08 | 67.88 | 72.60 | 1.32 | 22.74 | 27.27 | 40.27 | 1.38 | 1.21 | 1.30 | 已完成；015/016跨campaign |
| HAWQ-v3 FP4, latency target=.80 BF16 (PTQ) | 56.83 | 83.42 | 68.82 | 72.83 | 1.32 | 22.83 | 27.35 | 40.64 | 1.40 | 1.19 | 1.30 | 已完成；015/016跨campaign |
| HAWQ FP4 minimum-cost control, q=.07 | 55.89 | 81.52 | 69.61 | 71.67 | 2.19 | 22.66 | 27.11 | 39.49 | 2.20 | 1.57 | 1.87 | 已完成；015/016跨campaign |
| DASO, 5 candidates, fresh-profile DP | 53.58 | 80.89 | 66.69 | 69.04 | 1.68 | 21.36 | 24.58 | 37.39 | 1.95 | 1.85 | 1.90 | 已完成；015/016跨campaign |
| DASO + 8bit, fresh-profile DP | 57.76 | 82.66 | 68.59 | 71.77 | 1.98 | 22.62 | 26.71 | 39.63 | 1.79 | 1.40 | 1.59 | 已完成；015/016跨campaign |
| DASO + 8bit + exact/validation, epsilon=.01 | 57.17 | 83.54 | 68.98 | 72.61 | 1.89 | 22.45 | 26.87 | 39.58 | 1.69 | 1.38 | 1.55 | 已完成；015/016跨campaign |
| DASO + 8bit + exact/validation, epsilon=.04 | 56.06 | 81.02 | 69.22 | 70.34 | 2.21 | 21.67 | 25.44 | 38.58 | 1.97 | 1.57 | 1.78 | 已完成；015/016跨campaign |
| DASO-8 incl. plain INT8 + exact/validation, epsilon=0.01 | 56.91 | 82.74 | 67.88 | 72.33 | 1.86 | 22.82 | 26.96 | 40.13 | 1.00 | 1.00 | 1.00 | 已完成；015/016跨campaign |
| DASO-8 incl. plain INT8 + exact/validation, epsilon=0.04 | 55.89 | 81.52 | 69.61 | 71.67 | 2.17 | 21.72 | 25.07 | 38.39 | 2.26 | 1.83 | 2.05 | 已完成；015/016跨campaign |

## 表4 — Reviewer AK6a / Exi5 / LrT2 / gkP9：同候选精度分配与阶段统一（P0/P1）

当前数值直接来自各run冻结点CSV；空速度不借用校准结果。配对误差与同候选协议见[12组Llama诊断](night151/LLAMA_DIAGNOSTICS.md)。

共同候选为BF16/plain INT8/NVFP4；同成本、同上限、同MILP，只换DASO代理或Hessian代理。同算子：每模块两阶段方法相同；阶段适配：4bit用NVFP4-prefill/W4A16-decode。Prefill只需一组。每个格最终填“DASO / HAWQ”，报告真实ΔNLL、加速和目标违约，不能只放代理目标值。

Qwen九组速度/NLL已完成；Llama-3.1九组NLL与中心三组速度完成，其他速度明确待补；Llama-2中心三组完成，其他预算为P2。signed Hessian为预定主版本，[clip_zero/absolute的完整中心NLL](night151/HESSIAN_VARIANTS.md)另外保留，不能由signed结果推广为优于所有Hessian变体。

| 模型 | 场景 | 阶段模式 | r | 实测ΔNLL：D/H | E2E ×：D/H | 超时/总样本：D/H | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-2 | Prefill | 同算子 | .95 | 待 / 待 | 待 / 待 | 待 / 待 | P2扩展 |
| Llama-2 | Prefill | 同算子 | .80 | 0.00502 / 0.00569 | 1.318 / 1.326 | 0/24 / 0/24 | 三轮速度/NLL完成 |
| Llama-2 | Prefill | 同算子 | .65 | 待 / 待 | 待 / 待 | 待 / 待 | P2扩展 |
| Llama-2 | P+D | 同算子 | .95 | 待 / 待 | 待 / 待 | 待 / 待 | P2扩展 |
| Llama-2 | P+D | 同算子 | .80 | 0.01091 / 0.01461 | 1.316 / 1.318 | 0/24 / 0/24 | 三轮速度/NLL完成 |
| Llama-2 | P+D | 同算子 | .65 | 待 / 待 | 待 / 待 | 待 / 待 | P2扩展 |
| Llama-2 | P+D | 阶段适配 | .95 | 待 / 待 | 待 / 待 | 待 / 待 | P2扩展 |
| Llama-2 | P+D | 阶段适配 | .80 | 0.00110 / 0.00393 | 1.320 / 1.321 | 0/24 / 0/24 | 三轮速度/NLL完成 |
| Llama-2 | P+D | 阶段适配 | .65 | 待 / 待 | 待 / 待 | 待 / 待 | P2扩展 |
| Llama-3.1 | Prefill | 同算子 | .95 | 0.00930 / 0.00670 | 待 / 待 | 待 / 待 | NLL完成；速度待补 |
| Llama-3.1 | Prefill | 同算子 | .80 | 0.00908 / 0.00935 | 1.316 / 1.314 | 0/24 / 0/24 | 三轮速度/NLL完成 |
| Llama-3.1 | Prefill | 同算子 | .65 | 0.00860 / 0.01139 | 待 / 待 | 待 / 待 | NLL完成；速度待补 |
| Llama-3.1 | P+D | 同算子 | .95 | 0.00002 / 0.00742 | 待 / 待 | 待 / 待 | NLL完成；速度待补 |
| Llama-3.1 | P+D | 同算子 | .80 | 0.00698 / 0.00958 | 1.322 / 1.319 | 0/24 / 0/24 | 三轮速度/NLL完成 |
| Llama-3.1 | P+D | 同算子 | .65 | 0.01319 / 0.01812 | 待 / 待 | 待 / 待 | NLL完成；速度待补 |
| Llama-3.1 | P+D | 阶段适配 | .95 | 0.00090 / 0.00561 | 待 / 待 | 待 / 待 | NLL完成；速度待补 |
| Llama-3.1 | P+D | 阶段适配 | .80 | 0.00345 / 0.00794 | 1.317 / 1.330 | 0/24 / 0/24 | 三轮速度/NLL完成 |
| Llama-3.1 | P+D | 阶段适配 | .65 | 0.01499 / 0.01418 | 待 / 待 | 待 / 待 | NLL完成；速度待补 |
| Qwen3 | Prefill | 同算子 | .95 | 0.00512 / -0.00040 | 1.103 / 1.108 | 0/24 / 0/24 | 三轮速度/NLL完成 |
| Qwen3 | Prefill | 同算子 | .80 | 0.00641 / 0.00660 | 1.309 / 1.310 | 0/24 / 0/24 | 三轮速度/NLL完成 |
| Qwen3 | Prefill | 同算子 | .65 | 0.00976 / 0.00937 | 1.606 / 1.607 | 0/24 / 0/24 | 三轮速度/NLL完成 |
| Qwen3 | P+D | 同算子 | .95 | -0.00050 / 0.00629 | 1.106 / 1.105 | 0/24 / 0/24 | 三轮速度/NLL完成 |
| Qwen3 | P+D | 同算子 | .80 | 0.00089 / 0.01082 | 1.308 / 1.298 | 0/24 / 0/24 | 三轮速度/NLL完成 |
| Qwen3 | P+D | 同算子 | .65 | 0.01335 / 0.01216 | 1.591 / 1.591 | 0/24 / 0/24 | 三轮速度/NLL完成 |
| Qwen3 | P+D | 阶段适配 | .95 | 0.00167 / 0.00419 | 1.097 / 1.105 | 0/24 / 0/24 | 三轮速度/NLL完成 |
| Qwen3 | P+D | 阶段适配 | .80 | 0.00268 / 0.01125 | 1.296 / 1.305 | 0/24 / 0/24 | 三轮速度/NLL完成 |
| Qwen3 | P+D | 阶段适配 | .65 | 0.00446 / 0.00886 | 1.600 / 1.604 | 0/24 / 0/24 | 三轮速度/NLL完成 |

## 表5 — Reviewer AK6a / gkP9：质量预算是否真正满足（P0）

新增validation行按“validation guard / test均值±2SE / 是否超限”填写；其他原始代理行仍使用其代理预测，未测预算为—。20点的正式速度与选中map见[完整五预算表](responses/reviewer1/five_budget_llama_night151/README.md)。

本轮021五档预算20点均值0超限（共用原确认样本）；正式速度补充等022。另[026样本量跟进](responses/reviewer1/validation_size_run026/README.md)在固定14个PD incumbent上增加32个validation窗口，并用64个新窗口比较，不能混作原主表或全候选重新优化。

每个预算单元最终填`预测ΔNLL / confirmation实测ΔNLL±2SE / 是否超限`，另存超限幅度、选中policy和去重计数。不能将多个预算选中同一配置当成多个独立配置证明。

| 模型 | 场景 | 流程 | ε=.005 | ε=.01 | ε=.02 | ε=.04 | ε=.07 | 来源/待补 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-2 | Prefill | 原五候选DP | 待 | 待 | 待 | 待 | — | 原prefill实测NLL已归档，预算配对/新确认待补 |
| Llama-2 | Prefill | DASO-8代理精确解 | 待 | 待 | 待 | 待 | — | 待新增 |
| Llama-2 | Prefill | DASO-8 + validation | 0.0000 / 0.0000±0.0000 / 通过 | 0.0100 / 0.0027±0.0009 / 通过 | 0.0155 / 0.0087±0.0022 / 通过 | 0.0267 / 0.0242±0.0033 / 通过 | 0.0633 / 0.0494±0.0054 / 通过 | 021+022已完成；共用确认样本与重复map明确 |
| Llama-2 | P+D | 原五候选DP | 待 | 待 | 待 | 待 | — | 待补 |
| Llama-2 | P+D | DASO-8代理精确解 | 待 | 待 | 待 | 待 | — | 待新增 |
| Llama-2 | P+D | DASO-8 + validation | 0.0000 / 0.0000±0.0000 / 通过 | 0.0000 / 0.0000±0.0000 / 通过 | 0.0191 / 0.0099±0.0096 / 通过 | 0.0255 / 0.0043±0.0078 / 通过 | 0.0438 / 0.0182±0.0101 / 通过 | 021+022已完成；共用确认样本与重复map明确 |
| Llama-3.1 | Prefill | 原五候选DP | 待 | 待 | 待 | 待 | — | 原prefill NLL已归档，预算配对/新确认待补 |
| Llama-3.1 | Prefill | DASO-8代理精确解 | 待 | 待 | 待 | 待 | — | 待新增 |
| Llama-3.1 | Prefill | DASO-8 + validation | 0.0000 / 0.0000±0.0000 / 通过 | 0.0000 / 0.0000±0.0000 / 通过 | 0.0146 / 0.0084±0.0013 / 通过 | 0.0146 / 0.0084±0.0013 / 通过 | 0.0550 / 0.0453±0.0027 / 通过 | 021+022已完成；共用确认样本与重复map明确 |
| Llama-3.1 | P+D | 原五候选DP | 待 | 待 | 待 | 待 | — | 待补 |
| Llama-3.1 | P+D | DASO-8代理精确解 | 待 | 待 | 待 | 待 | — | 待新增 |
| Llama-3.1 | P+D | DASO-8 + validation | 0.0000 / 0.0000±0.0000 / 通过 | 0.0087 / 0.0015±0.0022 / 通过 | 0.0128 / 0.0070±0.0030 / 通过 | 0.0388 / 0.0284±0.0073 / 通过 | 0.0586 / 0.0320±0.0094 / 通过 | 021+022已完成；共用确认样本与重复map明确 |
| Qwen3 | Prefill | 七候选DP | 待汇总 | 待汇总 | 待汇总 | 待汇总 | — | run005/015已有；旧test窗口已在历史查看 |
| Qwen3 | Prefill | 七候选代理精确解 | 待汇总 | 待汇总 | 待汇总 | 待汇总 | — | run005/015已有 |
| Qwen3 | Prefill | 七候选 + validation | 待汇总 | 待汇总 | 待汇总 | 待汇总 | — | run005/015已有；不是016的新确认 |
| Qwen3 | P+D | 七候选DP | 待汇总 | 待汇总 | 待汇总 | 待汇总 | — | run005/015已有 |
| Qwen3 | P+D | 七候选代理精确解 | 待汇总 | 待汇总 | 待汇总 | 待汇总 | — | run005/015已有 |
| Qwen3 | P+D | 七候选 + validation | 待汇总 | 待汇总 | 待汇总 | 待汇总 | — | run005/015已有 |
| Qwen3 | Prefill | DASO-8代理精确解 | — | 0.0100 / 0.0404±0.0043 / 超 | — | 0.0394 / 0.1325±0.0100 / 超 | — | 016已完成；原始提案与validation选择分列 |
| Qwen3 | Prefill | DASO-8 + validation | — | -0.0016 / -0.0077±0.0040 / 通过 | — | 0.0204 / 0.0092±0.0048 / 通过 | — | 016已完成；原始提案与validation选择分列 |
| Qwen3 | P+D | DASO-8代理精确解 | — | 0.0099 / 0.0149±0.0181 / 超 | — | 0.0384 / 0.0076±0.0212 / 通过 | — | 016已完成；原始提案与validation选择分列 |
| Qwen3 | P+D | DASO-8 + validation | — | 0.0000 / 0.0000±0.0000 / 通过 | — | 0.0351 / 0.0076±0.0212 / 通过 | — | 016已完成；原始提案与validation选择分列 |

新增Llama先冻结四档预算和train/validation/confirmation划分；窗口配对比较BF16，最终test不用于选点。建议沿用已固定的validation `mean+2SE+.005`保护规则，并报告其准备成本。它是经验筛选，不是数学上的泛化保证；原DP超限及BF16回退均保留。精确求解只解决加和代理上的离散问题，不修复代理误差。

Qwen现有[run015质量约束CSV](runs/015/results/selected_test_constraints.csv)及[摘要](runs/015/results/constraints_summary.json)：10个质量预算点、6个唯一policy/scenario、实测均值0超限。此数字不是run016的新confirmation结论。

## 表6 — Reviewer gkP9 / AK6a：反向约束优化与实测时延违约（P1）

目标为给定延迟上限最小化质量损失；`r`为BF16校准E2E的比例。每个格分别填r=.95/.80/.65三个结果，不能只保留可行或有利的点。报告实测E2E与含恢复cycle两种超时计数；相同预测Linear上限不等于硬实时保证。

| 模型 | 场景 | r=.95：ΔNLL / E2E / 超时数 | r=.80：ΔNLL / E2E / 超时数 | r=.65：ΔNLL / E2E / 超时数 | cycle超时与不可行 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| Llama-2 | Prefill | 待 | 待 | 待 | 待 | P2扩展；优先中心点 |
| Llama-2 | P+D | 待 | 待 | 待 | 待 | P2扩展；优先中心点 |
| Llama-3.1 | Prefill | 待 | 待 | 待 | 待 | P1待补 |
| Llama-3.1 | P+D | 待 | 待 | 待 | 待 | P1待补 |
| Qwen3 | Prefill | 待汇总 | 待汇总 | 待汇总 | 待汇总 | run015已有反向选择；016另为同约束分配对照 |
| Qwen3 | P+D | 待汇总 | 待汇总 | 待汇总 | 待汇总 | run015已有反向选择；016运行中 |

现有来源：[run015冻结反向点](runs/015/results/selected_test_constraints.csv)、[早期反向与精确性检查](runs/005/REPORT.md)、[run012校准/test复核](runs/012/REPORT.md)。run015六个反向点全部可行、已测E2E/cycle均0超时；这与旧eager轮次P+D失败属于不同运行时证据，不能互相覆盖。

## 表7 — Reviewer Exi5 / LrT2 / QboE：有目的的B/S/O负载覆盖（P1）

019固定11标签/10map、六负载及每负载64窗口质量/三轮速度已完成；含O256，不按新负载重选。DASO-8 ε=.04的E2E数值最快5/6、cycle最快4/6，6个NLL均值均在预算内。见[完整结果](responses/reviewer2/workloads_run019/README.md)。

每个非中心case先比较固定主配置迁移，列BF16、SQ-INT8、FP8、HAWQ-FP4、硬件端点、DASO-8 ε=.01/.04七行；若做重新优化，单独标注并计入成本。每方法记录真实该长度ΔNLL与TTFT/TPOT/E2E/cycle，不能用中心质量替代。

| Case | B | S | O | 检验问题 | Qwen当前证据 | Llama待补范围 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 中心Prefill | 8 | 2048 | 1 | 对齐原Table2 | run015；016继续 | 两模型主表已有旧结果，统一重测 | P0 |
| 中心P+D | 8 | 2048 | 64 | 对齐原Table2 | run015；016继续 | 两模型主表统一重测 | P0 |
| 单请求长decode | 1 | 2048 | 128 | 小batch延迟/权重搬运 | run015 Graph已测速度，固定配置迁移 | Llama-3.1；Llama-2可共用设计 | P1 |
| 短prompt中等decode | 8 | 512 | 128 | 降低prefill占比 | run015 Graph已测速度，固定配置迁移 | Llama-3.1 | P1 |
| 长prompt | 8 | 4096 | 64 | prefill占比与长输入 | run015 Graph已测速度，固定配置迁移 | Llama-3.1；Llama-2用S3584，避免越过原上下文 | P1 |
| 长decode | 8 | 512 | 256 | 回答“只测64步” | run003有其他O256负载的旧eager；该统一Graph case待补 | Llama-3.1，必要时Llama-2 | P1 |
| 低batch长decode | 1 | 512 | 256 | 最强decode占比、暴露失败场景 | run003旧eager有；新Graph待补 | 先Llama-3.1，Llama-2为P2 | P1/P2 |
| 吞吐型Prefill | 16 | 2048 | 1 | 大batch计算受限 | run003旧eager有；新Graph待补 | 先Llama-3.1，先查KV/显存是否可执行 | P2 |

[run003完整16负载设计](runs/003/configs/design.json)是历史证据；[run015三种代表负载CSV](runs/015/results/representative_workloads.csv)包含66条归一化记录、两轮顺序。固定配置迁移不能称为逐负载重优化，也不自动带来逐负载质量保证。

## 表8 — Reviewer Exi5 / AK6a / gkP9：真实chunked/mixed调度与适用范围（P1，在线声明需更强测试）

下表每case对BF16、SQ-INT8、FP8、phase-uniform、HAWQ、DASO逐行报告。原global-phase单表示运行时与新增dual-resident逐token路由变体区分；同批必须记录真实mixed forward及每阶段token数。

| 模型/case | 到达模式 | chunk预算 | 必报质量 | 必报性能 | 必报成本 | 已有/缺口 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen B8/S2048/O64 | t0同时到达burst | 1024 | 该case teacher-forced ΔNLL | TTFT/TPOT/总完成时间/吞吐 | GPU/host驻留、路由耗时 | run006已完成旧变体八策略；新DASO-8待补 |
| Qwen B16/S512与4096交错/O64 | t0同时到达burst | 4096 | 同上 | 同上 | 同上 | run006已有；新DASO-8待补 |
| Qwen B8/S512与2048交错/O256 | t0同时到达burst | 1024 | 同上 | 同上 | 同上 | run006已有；新DASO-8待补 |
| Llama-3.1 B8/S2048/O64 | t0同时到达burst | 1024 | 新目标模型实测 | 同上 | 同上 | P1代表case待补 |
| Llama-3.1 异长prompt/O256 | t0同时到达burst | 1024 | 新目标模型实测 | 同上 | 同上 | P1；不需要Llama-2再复制整个矩阵 |
| Qwen + Llama-3.1 open-loop | 校准容量的25/50/75/90%，预先固定到达seed | 固定共同配置 | 同请求集质量 | TTFT/TPOT p50/p95/p99、RPS、超时/失败率 | 相同总GPU预算、KV/驻留限制、排队 | 条件扩展；当前未完成，不能声称已验证生产continuous serving |

[run006报告](runs/006/REPORT.md)是closed-loop burst，不是open-loop生产服务。若rebuttal将主结论限定为离线同阶段批次，可解释此边界并提供mixed变体补强；若保留生产SLA/continuous serving主张，最后一行就必须补，不能靠前五行代替。

## 表9 — Reviewer LrT2 / gkP9：延迟模型准确性、查表比较与非GEMM残差（P1）

本轮三模型六场景诊断已完成，实际点数、MAPE/p90/MAE与完整代理误差见[建模结果表](night151/MODELING_REPORT.md)。下面保留原Hybrid与计划项的区别，已测LUT不能冒充未知shape预测。

每组须报告样本数、MAPE/median/p90 APE、绝对MAE，并拆分kernel与完整policy；同一分区只用train拟合。原Table4的kernel MAPE已存在，但并不等于E2E预测误差。

| 模型/数据 | 预测方法 | 留出kernel MAPE | policy E2E误差 | Attention/KV/非Linear/恢复的处理 | 状态 |
| --- | --- | --- | --- | --- | --- |
| Llama-3.1原论文 | Direct fit | 14.80%（原文） | 待汇总原归档 | 原文原路径；另核查E2E数据 | 已有kernel，E2E待汇总 |
| Llama-3.1原论文 | + Roofline | 10.78%（原文） | 待汇总原归档 | 同上 | 同上 |
| Llama-3.1原论文 | + Shape correction | 8.01%（原文） | 待汇总原归档 | 同上 | 同上 |
| Llama-2新8bit路径 | Hybrid vs实测LUT | 待 | 待 | BF16非Linear单列；不能假定为0 | 新profile和代表policy待补 |
| Llama-3.1新8bit路径 | Hybrid vs实测LUT | 待 | 待 | 同上 | P1待补 |
| Qwen历史广shape库 | 原Hybrid，shape与N/K严格留出 | 已有：按五kernel分别4.18%–11.83% | 不能直接当新Graph E2E | 历史设备/路径，不能混用 | run004 broad数据已有 |
| Qwen新8bit/Graph路径 | Hybrid、实测LUT、Linear和+BF16 offset | 待针对新路径汇总 | 待对照run015/016实测 | 完整E2E、restore/cycle另列 | run015/016有原始cost与时间，分析待补 |

来源：[原模型消融](../../daso/results/paper/tables/ablation.tex)、[原E2E指标](../../daso/results/paper/ablations/data/modeling/e2e_latency_metrics.csv)、[run004广shape留出](runs/004/results/broad_latency_validation.csv)、[小样本失败留出](runs/004/results/latency_holdout_metrics.csv)、[run015实测画像](runs/015/results/graph_action_profile.csv)。小画像外推的高误差仍保留，不用广shape平均值掩盖。LUT只对已测shape可直接查；未知shape如何处理须说明，不拿oracle当免费可用模型。

## 表10 — Reviewer LrT2 / QboE / gkP9：完整准备、建模与搜索开销（P0/P1）

当前实际完成的叶子进程成本与缓存复制分列见[成本报告](night151/COST_REPORT.md)及[逐命令CSV](night151/preparation_processes.csv)。profile、两LlamaHessian/新特征/96新标签/拟合/validation/正式速度已完成；完整下游仍进行，未结束进程尚不计入，不能把快照当从零总成本。

同一单元分别列`wall time / 累计进程或GPU-active time / 复用缓存`；不同口径不能相加后称为wall或GPU-active。以目标模型从零成本与缓存复用成本分别报告。

| 阶段 | Llama-2 | Llama-3.1 | Qwen3 | 需保留的规模/说明 |
| --- | --- | --- | --- | --- |
| 目标硬件kernel profiling | 待整理原成本+新增8bit | 待整理原成本+新增8bit | run004/015已有，待汇总 | shape数、方法数、每GPU重测范围 |
| 激活与局部扰动采集 | 新8bit待测 | 新8bit待测 | run016 plain INT8已完成 | 样本/长度、融合模块数、特征缓存 |
| SQ scales与剪枝权重准备 | 新增基线待测 | 新增基线待测 | 已有002/007 | 模型载入/统计/量化/保存拆分 |
| Hessian敏感度采集 | 待新增 | 待新增 | 已有001，016复用需计原成本 | probe数、校准块、负trace处理 |
| 完整policy训练标签 | 目标模型八候选待补 | 目标模型八候选待补 | 原120/场景+新24/场景已采 | train/holdout数量、各标签窗口数 |
| 质量模型拟合 | 待 | 待 | run016两场景约4.69s/3.84s | 原结构/超参数；拟合与标签成本分开 |
| DP或MILP纯求解 | 待汇总+新候选 | 待汇总+新候选 | run005/016已有 | 动作数、预算数、bins/gap、CPU |
| validation真实质量 | 待 | 待 | run016 37策略/43场景已完成 | 32窗口，候选池大小及去重 |
| validation校准测速与选择 | 待 | 待 | run016 23策略已完成 | 导出、加载、Graph捕获不能漏算 |
| 权重导出/初始化/Graph捕获 | 待实测 | 待实测 | run015/016逐命令已有 | 独立wall与峰值内存 |
| confirmation/完整下游 | 待 | 待 | 016已完成 | 这是最终评测成本，不冒充在线推理开销 |
| 总准备成本/摊销请求数 | 待 | 待 | 待所有子项汇总 | 每硬件/模型/负载复用哪些阶段 |

来源：[run004准备成本](runs/004/results/preparation_costs.csv)、[run005新增validation成本](runs/005/REPORT.md)、[run016成本汇总代码](runs/016/code/preparation_costs.py)。原文54个质量标签、48.6 GPU-min是原消融的口径，不能拿它代表新增八候选全流程。

## 表11 — Reviewer AK6a / LrT2 / gkP9：阶段表示的GPU、host、传输与完整cycle（P1）

本轮三模型18行阶段表示/传输/cycle结果已完成：[完整表及统计口径](night151/RUNTIME_MEMORY_REPORT.md)。下列计划中的独立运行期NVML真实峰值未采，不能拿进程结束采样值填充；初始化、跨场景进程峰值与arena/host已在新表拆分。

以下每模型五行均须填实际测量。GPU峰值区分初始化与运行；host权重、pinned、RSS不能重复相加。固定同样KV容量，单表示异步复用与dual-resident分开。

| 模型 | 方法 | GPU权重GiB | GPU初始化峰值GiB | GPU运行峰值GiB | Host/pinned GiB | decode H2D GiB | restore ms | E2E / cycle ms | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-2 | BF16 | 待 | 待 | 待 | 0 / 0 | 0 | 待 | 待 / 待 | 待统一实测 |
| Llama-2 | SQ-INT8 | 待 | 待 | 待 | 待 | 待 | 待 | 待 / 待 | 待统一实测 |
| Llama-2 | HAWQ-FP4主表配置 | 待 | 待 | 待 | 待 | 待 | 待 | 待 / 待 | 待统一实测 |
| Llama-2 | DASO-8 ε=.01 | 待 | 待 | 待 | 待 | 待 | 待 | 待 / 待 | 待统一实测 |
| Llama-2 | DASO-8 ε=.04 | 待 | 待 | 待 | 待 | 待 | 待 | 待 / 待 | 待统一实测 |
| Llama-3.1 | BF16 | 待 | 待 | 待 | 0 / 0 | 0 | 待 | 待 / 待 | 待统一实测 |
| Llama-3.1 | SQ-INT8 | 待 | 待 | 待 | 待 | 待 | 待 | 待 / 待 | 待统一实测 |
| Llama-3.1 | HAWQ-FP4主表配置 | 待 | 待 | 待 | 待 | 待 | 待 | 待 / 待 | 待统一实测 |
| Llama-3.1 | DASO-8 ε=.01 | 待 | 待 | 待 | 待 | 待 | 待 | 待 / 待 | 待统一实测 |
| Llama-3.1 | DASO-8 ε=.04 | 待 | 待 | 待 | 待 | 待 | 待 | 待 / 待 | 待统一实测 |
| Qwen3 | BF16 | 待 | 待 | 待 | 0 / 0 | 0 | 待 | 待 / 待 | 015已有母版，待汇总 |
| Qwen3 | SQ-INT8 | 待 | 待 | 待 | 待 | 待 | 待 | 待 / 待 | 015已有母版，待汇总 |
| Qwen3 | HAWQ-FP4主表配置 | 待 | 待 | 待 | 待 | 待 | 待 | 待 / 待 | 015已有母版，待汇总 |
| Qwen3 | DASO-8 ε=.01 | 待 | 待 | 待 | 待 | 待 | 待 | 待 / 待 | 016运行中 |
| Qwen3 | DASO-8 ε=.04 | 待 | 待 | 待 | 待 | 待 | 待 | 待 / 待 | 016运行中 |

Qwen旧七候选/HAWQ对应原始数据：[run015 runtime_overhead.csv](runs/015/results/runtime_overhead.csv)。新八候选不能直接继承旧map峰值。双驻留的额外显存及新Graph初始化不应被省略；完整cycle含请求间恢复，E2E仍单独保留。

## 表12 — Reviewer gkP9 / LrT2：跨层格式转换与异步搬运控制（P1）

| 实验/接口 | 比较对象 | 输入/输出dtype与布局 | 必报指标 | Qwen已有来源 | Llama补充 |
| --- | --- | --- | --- | --- | --- |
| 同格式连续模块 | BF16→BF16、NVFP4→NVFP4、INT8→INT8 | 按真实边界记录，不按内部权重位宽推断 | 接口tensor SHA/shape/stride、耗时 | run004接口审计 | Llama-3.1代表层 |
| 异格式连续模块 | SparseNVFP4→BF16、NVFP4→INT8等真实相邻模块 | 多数接口为BF16，是否cast/repack据实测 | 显式转换次数/耗时、kernel内量化/metadata成本 | run004已有98接口 | 新8bit邻接补核验 |
| 同步 vs 异步 | 完全相同有效policy | 权重与输入一致 | 暴露H2D、TTFT/E2E/cycle、输出等价 | run004旧路径已有 | 新Graph代表policy待补 |
| arena复用 vs 双驻留 | 同一policy | 同数值权重、不同驻留策略 | GPU/host差值、恢复时间、Graph/分配影响 | run004/012旧控制已有 | Llama-3.1代表policy待补 |
| 相加预测 vs 全模型 | 同一目标模型与负载 | Attention/KV/非GEMM仍原路径 | Linear和、完整E2E、非Linear/调度残差 | run004 additive/trace；015新时间 | 两Llama代表policy |

来源：[接口CSV](runs/004/results/mlp_interfaces.csv)、[实际加和残差](runs/004/results/additive_whole_model.csv)、[run012控制](runs/012/REPORT.md)。kernel画像已经包含的激活量化/scale/metadata时间不应重复计费；实际没有发生的跨层weight repack也不能臆造为免费或非零。

## 表13 — Reviewer QboE / AK6a：简单剪枝与SparseGPT比较（P1）

Llama-3.1的四MC、完整word PPL及三生成/三轮速度已齐，见[完整剪枝表](responses/reviewer4/pruning_llama31_run020/README.md)。SG在MMLU/PPL领先，Wanda在部分生成分数更高；不将任何方法称为所有指标最优。

020已完成Llama-3.1 Wanda/magnitude/SparseGPT 2:4公平控制的完整MC、三生成、NLL和三轮速度，见[020](runs/020/README.md)。Llama-2新剪枝未开展，校准重建差1token的记录保留。

统一2:4 mask结构、同BF16 sparse kernel、同校准集（magnitude不需校准）。先完成全部四项MC+WikiText PPL；下表用MMLU/PPL作汇总入口，其余任务不省略。若补生成则按三任务完整协议另列，不用NLL替代任务分数。

| 模型 | 剪枝方法 | 校准块 | MMLU % | WikiText word PPL | Prefill TTFT × | P+D E2E × | 准备s | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-2 | BF16 | — | 46.446 | 12.228 | 1.000 | 1.000 | — | 018统一新路径已完成；其他剪枝仍待 |
| Llama-2 | SparseGPT 2:4 | 按原协议核对 | 28.251 | 19.869 | 1.599 | 1.134 | 待 | 018统一新路径已完成；其他剪枝仍待 |
| Llama-2 | Wanda 2:4 | 固定与SparseGPT相同 | 待 | 待 | 待 | 待 | 待 | 待补 |
| Llama-2 | Magnitude 2:4 | 0 | 待 | 待 | 待 | 待 | 待 | 待补 |
| Llama-3.1 | BF16 | — | 68.445 | 9.425 | 1.000 | 1.000 | — | 017/020完整质量与三轮速度已完成 |
| Llama-3.1 | SparseGPT 2:4 | 128 | 42.921 | 21.436 | 1.520 | 1.149 | 历史缓存；原计时未恢复 | 017/020完整质量与三轮速度已完成 |
| Llama-3.1 | Wanda 2:4 | 128 | 37.018 | 38.268 | 1.525 | 1.128 | 134.79 | 017/020完整质量与三轮速度已完成 |
| Llama-3.1 | Magnitude 2:4 | 0 | 28.564 | 516.328 | 1.528 | 1.130 | 31.61 | 017/020完整质量与三轮速度已完成 |
| Qwen3 | BF16 | — | 72.874 | 13.492 | 1.00 | 1.00 | — | 完整质量已有002/010 |
| Qwen3 | SparseGPT 2:4 | 128 | 54.102 | 20.125 | 1.53 | 1.08 | 305.03（旧准备记录） | 质量010、Graph速度015 |
| Qwen3 | Wanda 2:4 | 128 | 53.561 | 23.809 | 待Graph | 待Graph | 144.10（含载入/保存） | 质量007/010已有，旧eager速度保留 |
| Qwen3 | Magnitude 2:4 | 0 | 23.594 | 922.415 | 待Graph | 待Graph | 38.44（含载入/保存） | 同上 |

来源：[run007](runs/007/REPORT.md)、[run010完整likelihood](runs/010/results/likelihood_summary.csv)。相同稀疏kernel会给相近速度，重点是质量与准备成本；不能通过降低基线校准量、改变mask比例来制造优势。

## 表14 — Reviewer QboE / LrT2：大模型可扩展性（P1条件项，当前仍有真实缺口）

| 证据层级 | 模型/规模 | 实际权重/真实推理 | 质量 | 性能/成本 | 当前状态与结论范围 |
| --- | --- | --- | --- | --- | --- |
| 搜索规模 | 36/64/80层动作数 | 重复8B动作表，不是30B部署 | — | 约.03–.24s，按原记录汇总 | 005已有；只支持求解规模 |
| 大shape kernel | Qwen3-32B四类N/K×四M×七方法 | 合成权重112项真实forward | — | kernel延迟、峰值、可执行性 | 004已有；只支持kernel维度 |
| 完整30/32B模型 | BF16 | 待真实预训练权重及相同TP设置 | NLL+代表完整任务 | TTFT/TPOT/E2E、GPU/host峰值 | 未完成；需权重、显存、TP兼容 |
| 完整30/32B模型 | SQ-INT8 | 同上 | 同上 | 同上+准备成本 | 条件项 |
| 完整30/32B模型 | HAWQ-FP4 PTQ | 同上 | 同上 | 同上+Hessian/ILP成本 | 条件项 |
| 完整30/32B模型 | DASO-8 ε=.01 | 同上 | 同上 | 同上+建模/validation成本 | 条件项 |
| 完整30/32B模型 | DASO-8 ε=.04 | 同上 | 同上 | 同上 | 条件项 |

若资源允许，先固定一种32B、中心B/S/O与这五种方法，不铺满全部负载。所有方法用相同TP、KV与运行时，不能拿量化单卡和BF16多卡直接作同口径速度比。若不能完成，应明确保留大模型局限；shape/搜索不能写成“完成30B实验”。

## 表15 — Reviewer LrT2 / Exi5：原RTX PRO 6000跨平台表的新增baseline（P2条件扩展）

| 模型/硬件 | 方法 | Prefill四项Acc均值 | 三生成Score均值 | Prefill TTFT × | P+D TTFT × | TPOT × | E2E × | GPU峰值 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-2 / PRO6000 | BF16 | 原表已有 | 原表已有 | 1.00 | 1.00 | 1.00 | 1.00 | 待核 | 原Table3；新路径需重测 |
| Llama-2 / PRO6000 | SQ-INT8 | 待 | 待 | 待 | 待 | 待 | 待 | 待 | 可用该设备后补 |
| Llama-2 / PRO6000 | FP8 | 待 | 待 | 待 | 待 | 待 | 待 | 待 | 同上 |
| Llama-2 / PRO6000 | HAWQ-FP4 PTQ | 待 | 待 | 待 | 待 | 待 | 待 | 待 | 同设备重新profile/分配 |
| Llama-2 / PRO6000 | DASO-8 ε=.01 | 待 | 待 | 待 | 待 | 待 | 待 | 待 | 不能复制5090成本 |
| Llama-2 / PRO6000 | DASO-8 ε=.04 | 待 | 待 | 待 | 待 | 待 | 待 | 待 | 同上 |

来源：[原跨平台规范表](../../daso/results/paper/tables/cross_platform_pro6000_final.tex)。沿用原紧凑表时各分项结果仍须保留，平均分只用于与原表排版对齐，不作为质量约束。硬件暂不可用时不阻塞5090两Llama的P0主表。

## 表16 — Reviewer LrT2 / gkP9 / AK6a：质量代理、搜索及简单分配消融（P1/P2）

023的两Llama共16点DP/MILP网格复核、8项反约束代理检查已完成，见[完整结果](runs/023/REPORT.md)；质量代理同候选比较见表4，不能把精确代理求解等同真实质量保证。

| 对照 | 保持一致的条件 | 必报结果 | 已有证据 | 原Llama/新八候选待补 | 优先级 |
| --- | --- | --- | --- | --- | --- |
| Local / Structure / Phase质量模型 | 相同train/holdout、候选及标签数 | holdout MAE/RMSE、预算相关误差 | Llama-3.1原MAE .0571/.0468/.0413 | 新8bit是否改变误差分布；不能以旧五候选MAE代替 | P1 |
| 七候选 vs 八候选 | 相同原模型结构/超参数，plain INT8有真实标签 | 候选使用、质量/速度、标签成本 | Qwen016已完成 | 两Llama八候选校准；七候选中间版轻量NLL/速度即可 | P0/P1 |
| DP vs精确加和代理求解 | 相同cost/quality矩阵和预算 | 可行性、目标差、求解时间、网格误差 | Qwen005有16穷举/22 Pareto检查 | Llama-3.1代表budget复核；不等同真实最优证明 | P1 |
| 质量代理 vsHessian分配 | 同候选/成本/约束/求解器 | 见表4全部预算，不只取胜点 | Qwen016已完成 | Llama-3.1完整、Llama-2中心 | P0/P1 |
| Uniform phase-pair | NVFP4→W4A16、SQ-INT8→W4A16等固定map | 实测NLL/完整质量/速度/host成本 | Qwen005/006/015有控制 | Llama-3.1作为简单强对照，阶段表另列 | P1 |
| 局部扰动贪心/随机可行 | 同动作/真实预算/预先固定seed与搜索次数 | 已测质量–速度、评估成本 | 原论文附录有60 challengers/20 DASO点 | 原始审计待整理；需新增时先Llama-3.1 | P2 |
| NAS方法完整移植 | 必须可比动作、训练/搜索成本和执行路径 | 完整公平比较 | 尚未完成 | 非人工reviewer明确必跑项；不自动铺开新NAS工程 | P2 |

原质量模型仍为可分解的加和代理，训练完整policy不会使其表达任意交互。新模型要补真实目标模型标签；不得拿DASO更多候选与HAWQ更少候选的直接排名作为纯代理优势。

## 表17 — Reviewer QboE / gkP9 / Exi5 / LrT2：无需重跑大矩阵的文档与复核

| 条目 | 对应reviewer | 已有证据 | 最终交付/仍需动作 | 状态 |
| --- | --- | --- | --- | --- |
| Figure2公式/字体 | QboE | run008零字体矢量PDF/SVG | rebuttal附清晰图，修订版替换；原PDF保留 | 已完成图，需编入答复 |
| NAS/HAQ/HAWQ定位与引用 | gkP9、Exi5、LrT2 | run008 NAS说明、HAWQ移植审计 | 承认相关方法，说明DASO的模块/阶段/异构部署贡献；HAQ引用/讨论，不把HAWQ当作HAQ复现 | 待合并文字 |
| identical module inputs | gkP9 | run008 IDENTICAL_INPUTS、016误差采集 | 说明dense backbone输入复用、候选输出不向后传播、原decode M64近似与真实策略NLL区别 | 已有说明，需编入答复 |
| 离线/在线边界 | Exi5 | run006真实mixed变体与原运行时区别 | 不将burst或离线Graph主表称为生产continuous/open-loop验证 | 需明确写入 |
| 指标/数据身份 | 全部 | 原表source CSV、generation loader、run015 provenance | ARC-E acc与acc_norm分开；dsum确认为DialogSum，见下述文字纠正；每模型记录模板及SHA | 已核查，需统一答复文字 |
| 硬件与精度名称 | Exi5、LrT2 | runtime与权重格式归档 | INT4 weight-only/FP4 W4A4/FP8/INT8 kernel明确；不把软件可运行INT4写成硬件INT4 Tensor Core支持 | 需统一标签 |
| 所有证据与失败 | 全部 | 001–015封存，016保留命令/版本/失败 | 新Llama独立run目录，reviewer目录放最终证据与来源；所有数值可追溯 | 持续执行 |

文字纠正：部分旧汇总（如run010文字）将`dsum`称为XSum；实际[loader的load_dsum](../../references/pmpd_eval_kit/pmpd_eval.py)读取`dialogsum_repo/test.csv`或`knkarthick/dialogsum`，使用`dialogue`/`summary`字段，冻结1500条输入是DialogSum。此处纠正文字，不改封存的原报告/数值。

## 建议执行次序与最小完成条件

1. 完成当前Qwen016，不抢占其单卡正式测速。
2. 两个Llama的P0：原五候选锚点统一测速；补SQ-INT8、FP8、plain INT8、HAWQ-FP4和目标模型DASO-8 ε=.01/.04，填齐表1/2核心行及表5。原Balanced/Max speed map保留，不以新map冒充原提交。
3. 以Llama-3.1为主补表4、6–13的代表性控制；Llama-2优先主表、质量预算与最小中心复核，避免两个Llama复制所有Qwen历史调试矩阵。
4. 同步汇总已有预测/开销/剪枝/文档证据；补新执行路径确实缺少的测量。完整下游仅覆盖预定主表与代表预算，其他探索点先NLL/速度，不做所有交叉组合。
5. 再判断30B、PRO6000、open-loop等条件项；明确未完成的concern。最终答复按表0回填每位人工reviewer，保留全部预算、不利结果和实际适用范围。

P0完成的最低标准是：**两张Llama补全主表 + 一张Qwen主表 + 实测质量预算及来源齐全**。同候选/阶段对照、开销与代表负载用来解释收益来源。完整30B和生产在线服务若没有做，始终单列未解决，不用其他结果冒充。
