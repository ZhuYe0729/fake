# Hessian处理敏感性（R2 Exi5 / R3 LrT2）

中心r=.80、同候选/成本/延迟上限，各负值处理规则均在confirmation前固定。signed为事先指定主版本，clip_zero和absolute补充报告，不按test胜负替换主行。所有数字是同64窗口的ΔNLL与±2SE；未完成的变体速度/完整任务不据此补造。

| Run | Label | ΔNLL ±2SE | Predicted Linear ms | Limit ms |
| --- | --- | ---: | ---: | ---: |
| 017 | daso_prefill_only_same_operator_r0.8 | 0.00908 ± 0.00119 | 709.42 | 710.89 |
| 017 | hawq_signed_prefill_only_same_operator_r0.8 | 0.00935 ± 0.00138 | 710.51 | 710.89 |
| 017 | hawq_clip_zero_prefill_only_same_operator_r0.8 | 0.00403 ± 0.00122 | 710.69 | 710.89 |
| 017 | hawq_absolute_prefill_only_same_operator_r0.8 | 0.00397 ± 0.00057 | 710.84 | 710.89 |
| 017 | daso_prefill_decode_same_operator_r0.8 | 0.00698 ± 0.00296 | 1108.80 | 1109.78 |
| 017 | hawq_signed_prefill_decode_same_operator_r0.8 | 0.00958 ± 0.00366 | 1109.46 | 1109.78 |
| 017 | hawq_clip_zero_prefill_decode_same_operator_r0.8 | 0.00226 ± 0.00340 | 1109.65 | 1109.78 |
| 017 | hawq_absolute_prefill_decode_same_operator_r0.8 | 0.00547 ± 0.00277 | 1109.46 | 1109.78 |
| 017 | daso_prefill_decode_phase_pair_r0.8 | 0.00345 ± 0.00228 | 1109.61 | 1109.78 |
| 017 | hawq_signed_prefill_decode_phase_pair_r0.8 | 0.00794 ± 0.00370 | 1109.57 | 1109.78 |
| 017 | hawq_clip_zero_prefill_decode_phase_pair_r0.8 | 0.00663 ± 0.00342 | 1109.49 | 1109.78 |
| 017 | hawq_absolute_prefill_decode_phase_pair_r0.8 | 0.00547 ± 0.00277 | 1109.46 | 1109.78 |
| 018 | daso_prefill_only_same_operator_r0.8 | 0.00502 ± 0.00143 | 650.93 | 651.45 |
| 018 | hawq_signed_prefill_only_same_operator_r0.8 | 0.00569 ± 0.00198 | 650.71 | 651.45 |
| 018 | hawq_clip_zero_prefill_only_same_operator_r0.8 | 0.00250 ± 0.00127 | 651.40 | 651.45 |
| 018 | hawq_absolute_prefill_only_same_operator_r0.8 | 0.00240 ± 0.00090 | 651.41 | 651.45 |
| 018 | daso_prefill_decode_same_operator_r0.8 | 0.01091 ± 0.00762 | 950.14 | 950.21 |
| 018 | hawq_signed_prefill_decode_same_operator_r0.8 | 0.01461 ± 0.00976 | 949.98 | 950.21 |
| 018 | hawq_clip_zero_prefill_decode_same_operator_r0.8 | 0.00792 ± 0.00905 | 950.14 | 950.21 |
| 018 | hawq_absolute_prefill_decode_same_operator_r0.8 | 0.00702 ± 0.00691 | 950.05 | 950.21 |
| 018 | daso_prefill_decode_phase_pair_r0.8 | 0.00110 ± 0.00873 | 950.04 | 950.21 |
| 018 | hawq_signed_prefill_decode_phase_pair_r0.8 | 0.00393 ± 0.00873 | 949.87 | 950.21 |
| 018 | hawq_clip_zero_prefill_decode_phase_pair_r0.8 | 0.00356 ± 0.00743 | 950.19 | 950.21 |
| 018 | hawq_absolute_prefill_decode_phase_pair_r0.8 | 0.00702 ± 0.00691 | 950.05 | 950.21 |

Llama-3.1 Prefill上clip/absolute的均值损失低于signed，也低于当前同约束DASO代理点；因此不能将对signed的比较推广为击败所有Hessian变体。P+D、阶段模式和误差范围分别查看，不作全面最优断言。

[完整CSV](llama_hessian_variants.csv)
