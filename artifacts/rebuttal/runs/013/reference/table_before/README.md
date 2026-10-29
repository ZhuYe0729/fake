# Qwen3-8B：论文 Table 2 格式的现有证据汇总

状态：尚未全部填满，不能称为已完成的最终主表。此表只整理已有记录，不启动新实验、不修改已封存 runs。

速度统一取 run011 最后一轮 GPU0、B8/S2048、O1/O64、6GiB KV 的结果。未在该轮测量的速度留空，不混入 run001/002/003 的数字。
精度来自完整下游评测原始 JSON；连接精度与速度前核对完整 method-map。准确率与 ROUGE-L 为百分数，IWSLT 为 SacreBLEU；ARC-C 用 acc_norm，ARC-E 用 acc，与原 Table 2 口径对齐。

| Model | Method | ARC-C | ARC-E | Wino. | MMLU | Prefill TTFT x | CNN/DM | DialogSum | IWSLT | P+D TTFT x | P+D TPOT x | P+D E2E x |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | BF16 | 56.91 | 83.46 | 68.03 | 72.87 | 1.00 | 22.81 | 26.96 | 40.13 | 1.00 | 1.00 | 1.00 |
| Qwen3-8B | Dense NVFP4 | 54.61 | 82.37 | 67.96 | 70.97 | 待补 | 22.35 | 27.08 | 40.20 | 待补 | 待补 | 待补 |
| Qwen3-8B | W4A16 NVFP4 | 56.48 | 83.54 | 69.22 | 71.92 | 待补 | 22.55 | 27.44 | 40.12 | 待补 | 待补 | 待补 |
| Qwen3-8B | W4A16 INT4 Marlin | 52.65 | 80.09 | 65.35 | 71.16 | 待补 | 21.96 | 26.35 | 40.02 | 待补 | 待补 | 待补 |
| Qwen3-8B | Sparse BF16 | 44.45 | 73.91 | 64.25 | 54.10 | 待补 | 21.45 | 21.29 | 32.30 | 待补 | 待补 | 待补 |
| Qwen3-8B | Sparse NVFP4 | 28.16 | 53.87 | 55.41 | 30.80 | 待补 | 13.53 | 5.32 | 0.91 | 待补 | 待补 | 待补 |
| Qwen3-8B | INT8 W8A8 | 57.08 | 82.28 | 67.72 | 72.55 | 1.89 | 22.79 | 27.27 | 39.79 | 待补 | 待补 | 0.94 |
| Qwen3-8B | SmoothQuant INT8 | 57.17 | 83.54 | 68.98 | 72.61 | 1.88 | 22.70 | 27.18 | 39.32 | 待补 | 待补 | 0.93 |
| Qwen3-8B | FP8 W8A8 | 55.72 | 83.16 | 67.96 | 72.85 | 1.69 | 22.79 | 26.97 | 40.07 | 待补 | 待补 | 0.97 |
| Qwen3-8B | SmoothQuant FP8 | 57.00 | 83.71 | 68.43 | 72.83 | 待补 | 22.83 | 27.03 | 40.01 | 待补 | 待补 | 待补 |
| Qwen3-8B | HAWQ-v3 INT4, q=.01 (PTQ adaptation) | 待补 | 待补 | 待补 | 待补 | 1.72 | 待补 | 待补 | 待补 | 待补 | 待补 | 0.80 |
| Qwen3-8B | HAWQ-v3 FP4, q=.01 (PTQ adaptation) | 待补 | 待补 | 待补 | 待补 | 1.72 | 待补 | 待补 | 待补 | 待补 | 待补 | 1.09 |
| Qwen3-8B | HAWQ FP4 minimum-cost control, q=.07 | 待补 | 待补 | 待补 | 待补 | 2.18 | 22.49 | 27.30 | 40.18 | 待补 | 待补 | 1.20 |
| Qwen3-8B | DASO, original 5 candidates (representatives) | 54.44 | 81.40 | 66.69 | 68.96 | 待补 | 21.67 | 25.44 | 38.58 | 待补 | 待补 | 1.44 |
| Qwen3-8B | DASO + 8bit, original DP (representatives) | 56.23 | 81.99 | 67.64 | 71.51 | 1.97 | 22.52 | 26.80 | 39.49 | 待补 | 待补 | 1.16 |
| Qwen3-8B | DASO + 8bit + exact/validation, epsilon=.01 | 57.17 | 83.54 | 68.98 | 72.61 | 1.88 | 22.81 | 26.96 | 40.13 | 待补 | 待补 | 1.00 |
| Qwen3-8B | DASO + 8bit + exact/validation, epsilon=.04 | 55.03 | 81.99 | 69.93 | 71.10 | 2.19 | 22.60 | 26.72 | 39.54 | 待补 | 待补 | 1.37 |

## 缺项与使用说明

- 强化 HAWQ 的 INT4/FP4 分配点目前有 NLL 与速度，缺完整下游任务，不能拿旧 HAWQ 策略的任务分数代填。
- HAWQ minimum-cost control 的 P+D 与 run001 已评测 phase-uniform 策略 map 完全相同，生成质量继承该点；该控制点不能用来证明 Hessian 分配优于统一分配。Prefill 策略不同，精度仍待补。
- HAWQ 行的 q 是构造其延迟上限时配对的 DASO 预算标签，不是 HAWQ 自身满足的质量预算；还存在 minimum-cost clamp，不能据此声称两方法同一质量约束。
- 最新 run011 没有记录 P+D 首 token 时刻，不能从总时长推造 TTFT/TPOT，也不能把独立 Prefill-only 的 TTFT 填到不同 P+D 策略上。BF16 的两个 1.00 仅为归一化恒等值。
- 若保留全部旧 uniform 行，还需在同一测速协议补齐这些行；原 run002 的对应精度和速度记录均仍保留。
- DASO + 8bit 实际为原五候选加 SQ-INT8、FP8，共七候选；文件名 daso8 是历史 ID。原 DP 代表点与 exact/validation 改进版分别列出。
- exact/validation epsilon=.01 在 Prefill 选中 SQ-INT8，在 P+D 回退 BF16；epsilon=.04 两侧均为异构策略。二者均在完整下游评测前固定，不叫 Max-speed/Balanced 新最优点。
- run010 报告中的 ARC-E 为 normalized accuracy；此表读取原始 acc，避免混用 acc 与 acc_norm。
- run011 的首轮/最后一轮存在排名变化；此表记录其最终轮并在 provenance.json 保留各已用点首轮加速比。run012 已做局部重测，不能将此表单轮差距写成稳定显著领先。

## 精确缺项

- Dense NVFP4: prefill_ttft_speedup, pd_ttft_speedup, pd_tpot_speedup, pd_e2e_speedup
- W4A16 NVFP4: prefill_ttft_speedup, pd_ttft_speedup, pd_tpot_speedup, pd_e2e_speedup
- W4A16 INT4 Marlin: prefill_ttft_speedup, pd_ttft_speedup, pd_tpot_speedup, pd_e2e_speedup
- Sparse BF16: prefill_ttft_speedup, pd_ttft_speedup, pd_tpot_speedup, pd_e2e_speedup
- Sparse NVFP4: prefill_ttft_speedup, pd_ttft_speedup, pd_tpot_speedup, pd_e2e_speedup
- INT8 W8A8: pd_ttft_speedup, pd_tpot_speedup
- SmoothQuant INT8: pd_ttft_speedup, pd_tpot_speedup
- FP8 W8A8: pd_ttft_speedup, pd_tpot_speedup
- SmoothQuant FP8: prefill_ttft_speedup, pd_ttft_speedup, pd_tpot_speedup, pd_e2e_speedup
- HAWQ-v3 INT4, q=.01 (PTQ adaptation): arc_c, arc_e, wino, mmlu, cnn_dm, dialogsum, iwslt, pd_ttft_speedup, pd_tpot_speedup
- HAWQ-v3 FP4, q=.01 (PTQ adaptation): arc_c, arc_e, wino, mmlu, cnn_dm, dialogsum, iwslt, pd_ttft_speedup, pd_tpot_speedup
- HAWQ FP4 minimum-cost control, q=.07: arc_c, arc_e, wino, mmlu, pd_ttft_speedup, pd_tpot_speedup
- DASO, original 5 candidates (representatives): prefill_ttft_speedup, pd_ttft_speedup, pd_tpot_speedup
- DASO + 8bit, original DP (representatives): pd_ttft_speedup, pd_tpot_speedup
- DASO + 8bit + exact/validation, epsilon=.01: pd_ttft_speedup, pd_tpot_speedup
- DASO + 8bit + exact/validation, epsilon=.04: pd_ttft_speedup, pd_tpot_speedup

[机器可读表格](main_table.csv) · [逐项来源与哈希](provenance.json) · [LaTeX](main_table.tex) · [生成代码](build_table.py)
