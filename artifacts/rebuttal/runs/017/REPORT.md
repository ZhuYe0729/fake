# llama31_8b_instruct：原模型强基线与DASO-8

核心三轮速度与64窗口confirmation已完成。预定完整任务也已完成。

所有新速度统一RTX5090 GPU0/NUMA0、B8/S2048/O1或64、6GiB BF16 KV、按层arena复用及native decode Graph；三轮3warmup+8timed与BF16首尾锚点。原论文数值保留，以下是统一新路径重新评测。

完整MC统一batch8、3GiB BF16 KV，逐进程先执行B8/S2047完整prompt-logprobs内存/有限值门槛；ARC-C acc_norm、ARC-E/Wino/MMLU acc；CNN/DM1000、DialogSum1500为ROUGE-L%，IWSLT333为SacreBLEU。生成batch=8，使用原论文PMPD文字包装及模型默认BOS；总输入最多3840、greedy最多256新token允许EOS。完整生成全部本轮重评，固定O测速另行执行。

| Method | ARC-C | ARC-E | Wino | MMLU | Prefill TTFT × | CNN/DM | DialogSum | IWSLT | P+D TTFT × | TPOT × | E2E × |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BF16 | 55.72 | 82.24 | 73.88 | 68.44 | 1.00 | 19.34 | 13.48 | 10.70 | 1.00 | 1.00 | 1.00 |
| Dense NVFP4 | 51.19 | 78.96 | 71.98 | 63.96 | 1.78 | 16.57 | 9.11 | 10.61 | 1.78 | 1.25 | 1.50 |
| W4A16 NVFP4 Marlin | 53.67 | 80.98 | 73.09 | 66.19 | 1.00 | 17.44 | 9.93 | 10.53 | 1.00 | 2.01 | 1.28 |
| Sparse BF16 (SparseGPT) | 38.48 | 68.27 | 66.22 | 42.92 | 1.52 | 6.69 | 5.10 | 3.04 | 1.53 | 0.87 | 1.15 |
| Sparse NVFP4 (SparseGPT) | 26.62 | 46.46 | 54.70 | 23.73 | 1.98 | 4.27 | 0.47 | 0.02 | 1.97 | 1.38 | 1.66 |
| INT8 W8A8 | 55.55 | 81.57 | 73.95 | 67.66 | 1.89 | 19.58 | 14.01 | 10.89 | 1.92 | 1.51 | 1.72 |
| SmoothQuant INT8 | 55.46 | 81.86 | 73.88 | 67.67 | 1.90 | 19.50 | 12.94 | 10.71 | 1.92 | 1.54 | 1.73 |
| FP8 W8A8 | 54.78 | 82.32 | 74.11 | 67.74 | 1.72 | 19.72 | 14.64 | 10.68 | 1.73 | 0.81 | 1.16 |
| Uniform SQ-INT8 prefill / W4A16 decode | 55.46 | 81.86 | 73.88 | 67.67 | 1.90 | 19.84 | 13.46 | 10.52 | 1.92 | 1.81 | 1.87 |
| DASO-5 fixed original balanced | 52.05 | 80.47 | 72.38 | 66.27 | 1.61 | 18.65 | 11.85 | 10.64 | 1.65 | 1.41 | 1.53 |
| DASO-5 fixed original max_speed | 32.94 | 61.45 | 60.85 | 31.23 | 2.24 | 14.58 | 4.29 | 1.36 | 2.22 | 1.59 | 1.89 |
| HAWQ-FP4 PTQ r=.80 | 55.29 | 82.24 | 73.95 | 68.13 | 1.31 | 18.94 | 13.65 | 10.60 | 1.37 | 1.28 | 1.33 |
| Common hardware-minimum FP4/INT8 control | 54.44 | 81.10 | 73.80 | 66.48 | 2.28 | 18.83 | 12.26 | 10.54 | 2.29 | 1.73 | 2.01 |
| DASO-8 measured-LUT + exact/validation epsilon=0.01 | 55.72 | 82.24 | 73.88 | 68.44 | 1.00 | 19.64 | 13.02 | 10.85 | 1.56 | 1.10 | 1.32 |
| DASO-8 measured-LUT + exact/validation epsilon=0.04 | 55.46 | 81.86 | 73.88 | 67.67 | 1.90 | 18.83 | 12.26 | 10.54 | 2.29 | 1.73 | 2.01 |

## 比较范围与来源

- HAWQ为Hessian/ILP的PTQ适配；FP4候选prefill用NVFP4、decode用W4A16，8bit两阶段均plain INT8。原目标不可行时明确记录，不静默放宽。
- 原DASO-5 Balanced/Max speed保留原method-map；DASO-8新增目标模型误差/标签、精确加和代理求解与独立validation，不把所有收益归给原DP。
- 共同硬件端点仅在BF16/BF16、INT8/INT8、NVFP4-prefill/W4A16-decode三类动作内取最小预测成本，不是八候选独立阶段空间的全局最小E2E；不以此单点归因纯分配器优势。
- 原标签经过BF16/NVFP4采样协议检查；若失败，按预定规则重新计算全部原policy标签。数据来源和缓存决策见原始目录。
- confirmation与validation分开；benchmark以前有整体评估，不称为全新语料。均值预算与paired误差不构成形式化质量保证。
- 实测E2E不含请求间恢复，完整cycle及内存/初始化成本另存；三轮范围不是置信区间。
- 全部冻结点均有NLL；非核心速度控制若尚未测，CSV标为speed_pending，不能用于实测排名。先完成两模型核心主表再扩展。

[主表CSV](results/main_table.csv) · [全部点](results/allocation_points.csv) · [逐项来源](results/provenance.json) · [三轮范围](results/speed_ranges.csv)
