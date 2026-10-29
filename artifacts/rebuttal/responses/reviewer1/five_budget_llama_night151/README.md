# 原Llama五档质量预算与正式速度（R1 AK6a / R5 gkP9）

021规则与代码在任何017/018 confirmation前固定；同一冻结候选池，按validation ΔNLL+2SE+0.005筛选后取校准E2E最快点。BF16保护值为0。预算共享相同确认样本，重复选中的配置不是新独立实验。

正式速度来自017/018核心campaign或022补充campaign，各自使用同模型/同卡BF16首尾锚点归一化；全部GPU0/NUMA0、同输入B8/S2048/O1或64。跨campaign汇总明确保留来源，不伪称20行同轮。

| Run | Scene | ε | Policy | Validation guard | Test ΔNLL ±2SE | E2E × | Mean exceeds ε | Campaign |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| 017 | prefill_only | 0.005 | uniform_dense_bf16 | 0.00000 | 0.00000 ± 0.00000 | 1.000 | False | 017 |
| 017 | prefill_only | 0.010 | uniform_dense_bf16 | 0.00000 | 0.00000 ± 0.00000 | 1.000 | False | 017 |
| 017 | prefill_only | 0.020 | uniform_sq_int8 | 0.01462 | 0.00844 ± 0.00130 | 1.902 | False | 017 |
| 017 | prefill_only | 0.040 | uniform_sq_int8 | 0.01462 | 0.00844 ± 0.00130 | 1.902 | False | 017 |
| 017 | prefill_only | 0.070 | daso8_exact_prefill_only_q0.04 | 0.05497 | 0.04531 ± 0.00269 | 2.295 | False | 022/017 |
| 017 | prefill_decode | 0.005 | uniform_dense_bf16 | 0.00000 | 0.00000 ± 0.00000 | 1.000 | False | 017 |
| 017 | prefill_decode | 0.010 | daso_independent_prefill_decode_r0.8 | 0.00871 | 0.00152 ± 0.00224 | 1.317 | False | 017 |
| 017 | prefill_decode | 0.020 | daso_prefill_decode_same_operator_r0.8 | 0.01283 | 0.00698 ± 0.00296 | 1.322 | False | 017 |
| 017 | prefill_decode | 0.040 | hardware_minimum_prefill_decode_phase_pair | 0.03882 | 0.02845 ± 0.00733 | 2.009 | False | 017 |
| 017 | prefill_decode | 0.070 | daso8_exact_prefill_decode_q0.01 | 0.05862 | 0.03198 ± 0.00945 | 2.124 | False | 022/017 |
| 018 | prefill_only | 0.005 | uniform_dense_bf16 | 0.00000 | 0.00000 ± 0.00000 | 1.000 | False | 018 |
| 018 | prefill_only | 0.010 | uniform_fp8 | 0.00997 | 0.00268 ± 0.00089 | 1.700 | False | 018 |
| 018 | prefill_only | 0.020 | uniform_int8 | 0.01548 | 0.00874 ± 0.00215 | 1.849 | False | 018 |
| 018 | prefill_only | 0.040 | hardware_minimum_prefill_only_same_operator | 0.02672 | 0.02420 ± 0.00329 | 2.187 | False | 018 |
| 018 | prefill_only | 0.070 | daso8_exact_prefill_only_q0.01 | 0.06326 | 0.04937 ± 0.00535 | 2.210 | False | 022/018 |
| 018 | prefill_decode | 0.005 | uniform_dense_bf16 | 0.00000 | 0.00000 ± 0.00000 | 1.000 | False | 018 |
| 018 | prefill_decode | 0.010 | uniform_dense_bf16 | 0.00000 | 0.00000 ± 0.00000 | 1.000 | False | 018 |
| 018 | prefill_decode | 0.020 | phase_dense_nvfp4_sq_int8 | 0.01906 | 0.00986 ± 0.00964 | 1.510 | False | 022/018 |
| 018 | prefill_decode | 0.040 | uniform_int8 | 0.02555 | 0.00434 ± 0.00784 | 1.533 | False | 018 |
| 018 | prefill_decode | 0.070 | hardware_minimum_prefill_decode_phase_pair | 0.04384 | 0.01823 ± 0.01007 | 1.789 | False | 018 |

NLL预算不保证MC、ROUGE或BLEU；完整任务优先覆盖主表ε=.01/.04。经验保护和±2SE不构成任意分布泛化或多重比较保证。

[完整CSV](llama_five_budget_complete.csv) · [预登记选择及审计](../../../runs/021/README.md) · [补充速度](../../../runs/022/README.md)
