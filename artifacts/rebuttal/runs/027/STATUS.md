# 027 快速对照：已完成

四配置128条Graph pilot、新组合与全SQ-INT8的完整三项均已完成。最终结果见[完整报告](REPORT.md)。以下是已经归档的历史完整生成参考，002为旧eager、旧DASO点为015 Graph。新pilot四配置均重测同一128条/任务，不把历史结果冒充新输出。

| 配置 | CNN/DM1000 | DialogSum1500 | IWSLT333 |
| --- | ---: | ---: | ---: |
| sq_int8_w4a16 | 22.733 | 27.063 | 39.988 |
| uniform_sq_int8 | 22.697 | 27.179 | 39.322 |
| uniform_int8 | 22.787 | 27.275 | 39.795 |
| old_daso7_e001 | 22.451 | 26.871 | 39.579 |

历史上新组合三项都高于旧DASO点；相对全SQ-INT8，CNN/DM和IWSLT更高、DialogSum更低。此观察不代表当前Graph新测结论。

019已经完成的同64窗口NLL显示新组合ΔNLL=+0.02009，paired ±2SE=0.01376；它是固定阶段组合对照，不直接标记为满足ε=.01的DASO方案。

[冻结设计](configs/design.json) · [历史来源SHA](reference/historical_generation.json) · [实时状态](results/worker.json)

## 已完成的019三轮中心负载速度

| 配置 | E2E × | Cycle ×（含恢复） | TTFT ms | TPOT ms | Restore ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed_sq_int8_w4a16 | 1.741 | 1.569 | 611.145 | 8.359 | 124.319 |
| uniform_sq_int8 | 1.636 | 1.636 | 606.589 | 9.597 | 0.133 |
| graph_phase_dense_nvfp4_sq_int8 | 1.551 | 1.460 | 671.558 | 9.793 | 71.444 |

来源为019三轮自身BF16首尾锚点归一化；027新生成已完成；上面的历史完整分数仍单列保留，最终比较以REPORT.md为准。
