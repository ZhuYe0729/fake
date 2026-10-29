# llama2_7b_chat：原模型强基线与DASO-8

核心三轮速度与64窗口confirmation已完成。预定完整任务也已完成。

所有新速度统一RTX5090 GPU0/NUMA0、B8/S2048/O1或64、10GiB BF16 KV、按层arena复用及native decode Graph；三轮3warmup+8timed与BF16首尾锚点。原论文数值保留，以下是统一新路径重新评测。

完整MC统一batch8、10GiB BF16 KV，逐进程先执行B8/S2047完整prompt-logprobs内存/有限值门槛；ARC-C acc_norm、ARC-E/Wino/MMLU acc；CNN/DM1000、DialogSum1500为ROUGE-L%，IWSLT333为SacreBLEU。生成batch=4，使用原论文PMPD文字包装及模型默认BOS；总输入最多3840、greedy最多256新token允许EOS。完整生成全部本轮重评，固定O测速另行执行。

| Method | ARC-C | ARC-E | Wino | MMLU | Prefill TTFT × | CNN/DM | DialogSum | IWSLT | P+D TTFT × | TPOT × | E2E × |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BF16 | 43.26 | 72.81 | 68.27 | 46.45 | 1.00 | 23.60 | 21.61 | 19.44 | 1.00 | 1.00 | 1.00 |
| Dense NVFP4 | 42.92 | 71.51 | 67.56 | 44.98 | 1.80 | 23.57 | 21.48 | 17.23 | 1.80 | 1.11 | 1.38 |
| W4A16 NVFP4 Marlin | 42.92 | 71.72 | 68.19 | 45.80 | 0.99 | 23.73 | 21.77 | 18.85 | 0.99 | 1.58 | 1.21 |
| Sparse BF16 (SparseGPT) | 34.98 | 63.80 | 63.85 | 28.25 | 1.60 | 15.35 | 13.46 | 3.95 | 1.61 | 0.86 | 1.13 |
| Sparse NVFP4 (SparseGPT) | 23.55 | 37.16 | 51.62 | 23.25 | 2.00 | 9.66 | 11.11 | 0.73 | 2.00 | 1.20 | 1.51 |
| INT8 W8A8 | 43.52 | 72.52 | 68.67 | 45.73 | 1.85 | 23.66 | 21.75 | 16.87 | 1.87 | 1.29 | 1.53 |
| SmoothQuant INT8 | 43.43 | 73.11 | 68.90 | 46.20 | 1.86 | 23.56 | 21.32 | 18.20 | 1.88 | 1.29 | 1.54 |
| FP8 W8A8 | 43.17 | 72.39 | 67.40 | 46.37 | 1.70 | 23.61 | 21.80 | 19.05 | 1.71 | 0.83 | 1.13 |
| Uniform SQ-INT8 prefill / W4A16 decode | 43.43 | 73.11 | 68.90 | 46.20 | 1.86 | 23.90 | 21.66 | 18.47 | 1.85 | 1.42 | 1.62 |
| DASO-5 fixed original balanced | 42.58 | 71.76 | 67.72 | 44.18 | 1.85 | 23.80 | 21.39 | 21.12 | 1.37 | 1.20 | 1.28 |
| DASO-5 fixed original max_speed | 28.24 | 53.32 | 54.93 | 23.66 | 2.25 | 18.46 | 13.57 | 2.61 | 2.24 | 1.51 | 1.81 |
| HAWQ-FP4 PTQ r=.80 | 42.49 | 71.89 | 68.03 | 45.88 | 1.33 | 23.84 | 21.60 | 18.12 | 1.43 | 1.22 | 1.32 |
| Common hardware-minimum FP4/INT8 control | 43.43 | 71.55 | 66.85 | 44.96 | 2.19 | 23.67 | 21.57 | 17.40 | 2.19 | 1.50 | 1.79 |
| DASO-8 measured-LUT + exact/validation epsilon=0.01 | 43.17 | 72.39 | 67.40 | 46.37 | 1.70 | 23.60 | 21.61 | 19.44 | 1.00 | 1.00 | 1.00 |
| DASO-8 measured-LUT + exact/validation epsilon=0.04 | 43.43 | 71.55 | 66.85 | 44.96 | 2.19 | 23.66 | 21.75 | 16.87 | 1.87 | 1.29 | 1.53 |

## 比较范围与来源

- HAWQ为Hessian/ILP的PTQ适配；FP4候选prefill用NVFP4、decode用W4A16，8bit两阶段均plain INT8。原目标不可行时明确记录，不静默放宽。
- 原DASO-5 Balanced/Max speed保留原method-map；DASO-8新增目标模型误差/标签、精确加和代理求解与独立validation，不把所有收益归给原DP。
- 共同硬件端点仅在BF16/BF16、INT8/INT8、NVFP4-prefill/W4A16-decode三类动作内取最小预测成本，不是八候选独立阶段空间的全局最小E2E；不以此单点归因纯分配器优势。
- 原标签经过BF16/NVFP4采样协议检查；若失败，按预定规则重新计算全部原policy标签。数据来源和缓存决策见原始目录。
- confirmation与validation分开；benchmark以前有整体评估，不称为全新语料。均值预算与paired误差不构成形式化质量保证。
- 实测E2E不含请求间恢复，完整cycle及内存/初始化成本另存；三轮范围不是置信区间。
- 全部冻结点均有NLL；非核心速度控制若尚未测，CSV标为speed_pending，不能用于实测排名。先完成两模型核心主表再扩展。

[主表CSV](main_table.csv) · [全部点](allocation_points.csv) · [逐项来源](provenance.json) · [三轮范围](speed_ranges.csv)

[原始代码、数据、日志与配置](../../../runs/018/README.md)
