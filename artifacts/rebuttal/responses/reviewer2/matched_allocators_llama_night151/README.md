# 原Llama：同候选分配器与独立预算确认（R1 AK6a / R2 Exi5 / R5 gkP9）

同候选对照使用共同BF16/INT8/FP4动作、同一实测Linear LUT、相同加和延迟上限与MILP求解规则。只替换质量目标：DASO校准代理或HAWQ signed Hessian PTQ代理。阶段统一/阶段pair分别比较；不将八候选独立阶段空间的收益全部归因于代理本身。

以下列出所有冻结的signed中心/扩展预算对照；尚未完成正式测速的点保留质量结果并明确标记。每个差值按同64窗口配对；±2SE是近似误差描述，未作多重比较校正，不据均值差宣称所有点显著优胜。

| Run | Scene | Phase mode | r | DASO ΔNLL | HAWQ ΔNLL | DASO−HAWQ ±2SE | DASO E2E × | HAWQ E2E × |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 017 | prefill_only | same_operator | 0.95 | 0.00930 | 0.00670 | 0.00259 ± 0.00140 | 待测速 | 待测速 |
| 017 | prefill_only | same_operator | 0.80 | 0.00908 | 0.00935 | -0.00026 ± 0.00151 | 1.316 | 1.314 |
| 017 | prefill_only | same_operator | 0.65 | 0.00860 | 0.01139 | -0.00279 ± 0.00131 | 待测速 | 待测速 |
| 017 | prefill_decode | same_operator | 0.95 | 0.00002 | 0.00742 | -0.00740 ± 0.00353 | 待测速 | 待测速 |
| 017 | prefill_decode | same_operator | 0.80 | 0.00698 | 0.00958 | -0.00260 ± 0.00417 | 1.322 | 1.319 |
| 017 | prefill_decode | same_operator | 0.65 | 0.01319 | 0.01812 | -0.00492 ± 0.00556 | 待测速 | 待测速 |
| 017 | prefill_decode | phase_pair | 0.95 | 0.00090 | 0.00561 | -0.00471 ± 0.00314 | 待测速 | 待测速 |
| 017 | prefill_decode | phase_pair | 0.80 | 0.00345 | 0.00794 | -0.00449 ± 0.00361 | 1.317 | 1.330 |
| 017 | prefill_decode | phase_pair | 0.65 | 0.01499 | 0.01418 | 0.00081 ± 0.00609 | 待测速 | 待测速 |
| 018 | prefill_only | same_operator | 0.80 | 0.00502 | 0.00569 | -0.00066 ± 0.00219 | 1.318 | 1.326 |
| 018 | prefill_decode | same_operator | 0.80 | 0.01091 | 0.01461 | -0.00369 ± 0.01154 | 1.316 | 1.318 |
| 018 | prefill_decode | phase_pair | 0.80 | 0.00110 | 0.00393 | -0.00283 ± 0.01127 | 1.320 | 1.321 |

## 原池validation选择的主表预算点

选择仅用validation的ΔNLL+2SE+0.005及校准速度，BF16保护值为0；下表报告独立test结果。这是新增DASO-8的经验保护流程，不能说原DP天然保证任意数据/任务质量；各预算共享确认样本。

| Run | Scene | ε | Selected policy | Validation guard | Test ΔNLL ±2SE | E2E × | Mean exceeds ε |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| 017 | prefill_only | 0.01 | uniform_dense_bf16 | 0.00000 | 0.00000 ± 0.00000 | 1.000 | False |
| 017 | prefill_only | 0.04 | uniform_sq_int8 | 0.01462 | 0.00844 ± 0.00130 | 1.902 | False |
| 017 | prefill_decode | 0.01 | daso_independent_prefill_decode_r0.8 | 0.00871 | 0.00152 ± 0.00224 | 1.317 | False |
| 017 | prefill_decode | 0.04 | hardware_minimum_prefill_decode_phase_pair | 0.03882 | 0.02845 ± 0.00733 | 2.009 | False |
| 018 | prefill_only | 0.01 | uniform_fp8 | 0.00997 | 0.00268 ± 0.00089 | 1.700 | False |
| 018 | prefill_only | 0.04 | hardware_minimum_prefill_only_same_operator | 0.02672 | 0.02420 ± 0.00329 | 2.187 | False |
| 018 | prefill_decode | 0.01 | uniform_dense_bf16 | 0.00000 | 0.00000 ± 0.00000 | 1.000 | False |
| 018 | prefill_decode | 0.04 | uniform_int8 | 0.02555 | 0.00434 ± 0.00784 | 1.533 | False |

质量预算针对WikiText的教师强制NLL；完整生成与MC另测，不能用本表代替。速度为三轮各自BF16首尾锚点归一化后中位数，均在GPU0/NUMA0；非核心点未借用calibration数值冒充正式速度。

[同候选逐点CSV](llama_matched_allocators.csv) · [预算CSV](llama_verified_budgets.csv) · [冻结来源SHA](llama_diagnostics_sources.json)
