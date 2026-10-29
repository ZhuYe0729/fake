# Llama-3.1：同候选分配与反向约束（R2 Exi5 / R5 gkP9）

原017全部18个固定map，BF16/INT8/NVFP4候选、同Linear上限与MILP，只改变质量代理。HAWQ为signed Hessian/ILP的PTQ硬件移植；不称CNN QAT/TVM原栈。NLL按SHA复用原64窗口，质量不重新拟合；全部有利/不利点保留。

统一GPU6/NUMA1，B8/S2048/O1或64，6GiB KV、native decode Graph，三轮各三warmup/八timed，BF16首尾锚点。表中三轮中位数，范围是重复范围。原绝对E2E上限不修改；current-BF16比例诊断与cycle恢复诊断另列CSV，不构成新SLA保证。

| 场景/阶段 | r | 分配器 | ΔNLL | ±2SE | E2E ms | ×BF16 | 三轮范围 | 原目标ms | E2E超限/24 | Cycle ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| prefill_only/same_operator | 0.95 | daso | 0.00930 | 0.00094 | 998.93 | 1.112 | 1.110–1.112 | 1057.83 | 0 | 999.03 |
| prefill_only/same_operator | 0.95 | hawq_signed | 0.00670 | 0.00132 | 996.85 | 1.114 | 1.113–1.115 | 1057.83 | 0 | 996.99 |
| prefill_only/same_operator | 0.80 | daso | 0.00908 | 0.00119 | 827.99 | 1.341 | 1.340–1.342 | 890.81 | 0 | 828.07 |
| prefill_only/same_operator | 0.80 | hawq_signed | 0.00935 | 0.00138 | 830.33 | 1.338 | 1.338–1.340 | 890.81 | 0 | 830.47 |
| prefill_only/same_operator | 0.65 | daso | 0.00860 | 0.00127 | 662.72 | 1.676 | 1.674–1.677 | 723.78 | 0 | 662.85 |
| prefill_only/same_operator | 0.65 | hawq_signed | 0.01139 | 0.00160 | 663.07 | 1.672 | 1.670–1.677 | 723.78 | 0 | 663.21 |
| prefill_decode/same_operator | 0.95 | daso | 0.00002 | 0.00146 | 1741.96 | 1.115 | 1.114–1.130 | 1876.37 | 0 | 1742.09 |
| prefill_decode/same_operator | 0.95 | hawq_signed | 0.00742 | 0.00367 | 1754.33 | 1.120 | 1.106–1.132 | 1876.37 | 0 | 1754.43 |
| prefill_decode/same_operator | 0.80 | daso | 0.00698 | 0.00296 | 1458.61 | 1.350 | 1.322–1.362 | 1580.10 | 0 | 1458.76 |
| prefill_decode/same_operator | 0.80 | hawq_signed | 0.00958 | 0.00366 | 1463.13 | 1.336 | 1.327–1.360 | 1580.10 | 0 | 1463.28 |
| prefill_decode/same_operator | 0.65 | daso | 0.01319 | 0.00511 | 1185.75 | 1.657 | 1.654–1.658 | 1283.83 | 0 | 1185.92 |
| prefill_decode/same_operator | 0.65 | hawq_signed | 0.01812 | 0.00455 | 1161.88 | 1.671 | 1.656–1.700 | 1283.83 | 0 | 1162.03 |
| prefill_decode/phase_pair | 0.95 | daso | 0.00090 | 0.00164 | 1741.80 | 1.115 | 1.114–1.130 | 1876.37 | 0 | 1741.94 |
| prefill_decode/phase_pair | 0.95 | hawq_signed | 0.00561 | 0.00341 | 1766.69 | 1.115 | 1.112–1.116 | 1876.37 | 0 | 1777.07 |
| prefill_decode/phase_pair | 0.80 | daso | 0.00345 | 0.00228 | 1471.71 | 1.335 | 1.318–1.348 | 1580.10 | 0 | 1471.84 |
| prefill_decode/phase_pair | 0.80 | hawq_signed | 0.00794 | 0.00370 | 1476.07 | 1.334 | 1.313–1.357 | 1580.10 | 0 | 1485.77 |
| prefill_decode/phase_pair | 0.65 | daso | 0.01499 | 0.00501 | 1161.55 | 1.671 | 1.656–1.697 | 1283.83 | 0 | 1162.86 |
| prefill_decode/phase_pair | 0.65 | hawq_signed | 0.01418 | 0.00523 | 1184.63 | 1.662 | 1.634–1.695 | 1283.83 | 0 | 1195.00 |

[完整18点](matched_points.csv) · [三轮明细](speed_rounds.csv) · [配对代理比较](paired_allocators.csv) · [原配置](design.json) · [审计](provenance.json)
