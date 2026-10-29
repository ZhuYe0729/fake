# Qwen3同候选分配器与阶段适配对照

三轮同卡速度与新confirmation窗口结果已完成。预定完整任务也已完成。

所有主对照共用BF16、plain INT8、NVFP4，三种延迟目标及同一MILP。两者分别最小化DASO质量代理或Hessian加权权重误差；并列解用同样的最小预测延迟规则。HAWQ为Hessian/ILP的PTQ移植，不称为原CNN QAT/TVM全栈复现。prefill/decode同算子与阶段后端适配均在评估前固定，现有run015强对照继续保留。

validation32和confirmation64与历史对应64窗口不重叠，但benchmark以前已有整体评估，不能称为全新语料。主负载B8/S2048/O1或64，GPU0/NUMA0 CPU0–7、6GiB BF16 KV、native decode Graph；三轮3warmup+8timed、BF16首尾锚点。表中为三轮归一化加速比中位数，范围为重复最小/最大值，不是置信区间。图中横线为窗口paired ±2SE的描述性误差条。恢复另列cycle。

## 同候选与同预测延迟上限

| Scenario / mode | Allocator | Target / BF16 | Confirmation ΔNLL | Paired ±2SE | E2E × | Campaign range | Target misses /24 |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| prefill_only/same_operator | daso matched | 0.95 | 0.00512 | 0.00118 | 1.103 | 1.101–1.104 | 0 |
| prefill_only/same_operator | hawq matched | 0.95 | -0.00040 | 0.00190 | 1.108 | 1.107–1.108 | 0 |
| prefill_only/same_operator | daso matched | 0.80 | 0.00641 | 0.00163 | 1.309 | 1.303–1.309 | 0 |
| prefill_only/same_operator | hawq matched | 0.80 | 0.00660 | 0.00277 | 1.310 | 1.308–1.312 | 0 |
| prefill_only/same_operator | daso matched | 0.65 | 0.00976 | 0.00213 | 1.606 | 1.604–1.608 | 0 |
| prefill_only/same_operator | hawq matched | 0.65 | 0.00937 | 0.00303 | 1.607 | 1.604–1.612 | 0 |
| prefill_decode/same_operator | daso matched | 0.95 | -0.00050 | 0.00403 | 1.106 | 1.099–1.107 | 0 |
| prefill_decode/same_operator | hawq matched | 0.95 | 0.00629 | 0.00649 | 1.105 | 1.100–1.107 | 0 |
| prefill_decode/same_operator | daso matched | 0.80 | 0.00089 | 0.00696 | 1.308 | 1.292–1.311 | 0 |
| prefill_decode/same_operator | hawq matched | 0.80 | 0.01082 | 0.00940 | 1.298 | 1.296–1.312 | 0 |
| prefill_decode/same_operator | daso matched | 0.65 | 0.01335 | 0.01232 | 1.591 | 1.580–1.607 | 0 |
| prefill_decode/same_operator | hawq matched | 0.65 | 0.01216 | 0.01123 | 1.591 | 1.588–1.610 | 0 |
| prefill_decode/phase_pair | daso matched | 0.95 | 0.00167 | 0.00501 | 1.097 | 1.090–1.108 | 0 |
| prefill_decode/phase_pair | hawq matched | 0.95 | 0.00419 | 0.00626 | 1.105 | 1.089–1.106 | 0 |
| prefill_decode/phase_pair | daso matched | 0.80 | 0.00268 | 0.00607 | 1.296 | 1.290–1.321 | 0 |
| prefill_decode/phase_pair | hawq matched | 0.80 | 0.01125 | 0.01042 | 1.305 | 1.292–1.307 | 0 |
| prefill_decode/phase_pair | daso matched | 0.65 | 0.00446 | 0.01177 | 1.600 | 1.600–1.627 | 0 |
| prefill_decode/phase_pair | hawq matched | 0.65 | 0.00886 | 0.01066 | 1.604 | 1.587–1.610 | 0 |
| prefill_decode/independent | daso independent matched formats | 0.95 | 0.00032 | 0.00669 | 1.259 | 1.255–1.269 | 0 |
| prefill_decode/independent | daso independent matched formats | 0.80 | 0.00237 | 0.00665 | 1.304 | 1.301–1.315 | 0 |
| prefill_decode/independent | daso independent matched formats | 0.65 | 0.00685 | 0.01147 | 1.599 | 1.595–1.617 | 0 |

## 八候选扩展的validation选择

该部分包含新增plain INT8训练与候选池校验，不能单独归因为分配器优于HAWQ。

| Scenario | ε | Confirmation ΔNLL | Mean violation | E2E × | Policy |
| --- | ---: | ---: | --- | ---: | --- |
| prefill_only | 0.01 | -0.00770 | False | 1.863 | uniform_int8 |
| prefill_only | 0.04 | 0.00916 | False | 2.167 | hardware_minimum_prefill_only_same_operator |
| prefill_decode | 0.01 | 0.00000 | False | 1.000 | uniform_dense_bf16 |
| prefill_decode | 0.04 | 0.00756 | False | 2.051 | daso8_exact_prefill_decode_q0.04 |

## 预定完整任务指标

中心.80对照与两个质量预算选择点全部列出，不按分数选择展示对象。Prefill为完整zero-shot ARC-C/ARC-E/Wino/MMLU；生成CNN/DM1000、DialogSum1500、IWSLT333，native Qwen3 non-thinking、greedy、最多256新token允许EOS。

| Label | Scenario | Metrics |
| --- | --- | --- |
| daso_prefill_only_same_operator_r0.8 | prefill_only | arc_c=55.97, arc_e=83.92, wino=68.11, mmlu=72.51 |
| hawq_signed_prefill_only_same_operator_r0.8 | prefill_only | arc_c=56.83, arc_e=83.42, wino=68.82, mmlu=72.83 |
| daso_prefill_decode_same_operator_r0.8 | prefill_decode | cnn_dm=22.75, dialogsum=26.96, iwslt=40.24 |
| hawq_signed_prefill_decode_same_operator_r0.8 | prefill_decode | cnn_dm=22.96, dialogsum=27.24, iwslt=40.63 |
| daso_prefill_decode_phase_pair_r0.8 | prefill_decode | cnn_dm=22.70, dialogsum=27.04, iwslt=40.22 |
| hawq_signed_prefill_decode_phase_pair_r0.8 | prefill_decode | cnn_dm=22.83, dialogsum=27.35, iwslt=40.64 |
| daso_independent_prefill_decode_r0.8 | prefill_decode | cnn_dm=22.69, dialogsum=27.16, iwslt=40.35 |
| daso8_verified_prefill_only_q0.01 | prefill_only | arc_c=56.91, arc_e=82.74, wino=67.88, mmlu=72.33 |
| daso8_verified_prefill_only_q0.04 | prefill_only | arc_c=55.89, arc_e=81.52, wino=69.61, mmlu=71.67 |
| daso8_verified_prefill_decode_q0.01 | prefill_decode | cnn_dm=22.82, dialogsum=26.96, iwslt=40.13 |
| daso8_verified_prefill_decode_q0.04 | prefill_decode | cnn_dm=21.72, dialogsum=25.07, iwslt=38.39 |

## 解释边界

- 速度为固定batch/长度的离线负载；不据此外推异步到达、连续批处理或在线服务SLA。完整生成指标在本轮运行时重新计算。
- 同一预测Linear上限不等于相同实测E2E；目标违约、恢复cycle和真实质量都保留，不能只比较名义r或ε。
- 原始signed Hessian为主对照；clip_zero/absolute只在预定中心预算作敏感性控制。硬件最小成本端点单列，并非Hessian模型胜出的证据。
- 共同使用实际融合权重的量化扰动，NV4/W4共享canonical表示；DASO第八动作有独立采集的误差/训练标签，没有复制SQ系数。
- 质量模型结构和超参数沿用原实现；本实验共用MILP，不能据此声称原DP离散实现已被证明最优。
- 计算成本包含新增训练标签和validation搜索；旧Hessian/原标签属于有成本的缓存复用，不能当成从零免费。

[全部预登记点](results/allocation_points.csv) · [选择点](results/selected_quality_points.csv) · [质量–速度图](results/matched_frontier.pdf) · [冻结设计](configs/frozen_design.json) · [全部原始代码与日志](README.md)
