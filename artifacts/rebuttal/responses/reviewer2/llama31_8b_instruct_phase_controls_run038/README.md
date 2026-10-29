# llama31_8b_instruct：统一8bit-prefill/W4A16-decode控制

固定两种既有全局阶段map，与已有SQ-prefill/W4-decode及uniform点比较；所有128模块的prefill分别为plain INT8或FP8，decode均W4A16。不是DASO求解器新选点，也未改精度模型。

本轮无代理预算/权重调参；NLL只作质量诊断，无strictε或优化器获胜声明。全部固定控制保留，原Prefill-only/原主表不改，无新增MC。

速度统一GPU6/NUMA1，B8/S2048/O64，原KV6GiB、native decode Graph；三轮3warmup+5timed、BF16首尾锚点、同轮INT8。E2E计入prefill→decode切换，不计下一批恢复。

完整生成保留原PMPD/BOS、batch8、greedy最多256新token/EOS；1000/1500/333全样本。相同map/源码/输入协议的017结果按SHA复用，其他新测。Avg是ROUGE-L%/BLEU三项算术摘要，不是统一准确率。

| 设置 | CNN/DM | DialogSum | IWSLT | Gen Avg | TTFT ms | TPOT ms | E2E ms | ×BF16 | ×INT8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform_dense_bf16 | 19.337 | 13.482 | 10.697 | 14.505 | 1105.538 | 13.468 | 1953.117 | 1.000 | 0.567 |
| uniform_int8 | 19.585 | 14.011 | 10.889 | 14.828 | 549.909 | 8.848 | 1107.270 | 1.764 | 1.000 |
| uniform_sq_int8 | 19.505 | 12.942 | 10.709 | 14.385 | 549.654 | 8.987 | 1115.933 | 1.750 | 0.992 |
| uniform_fp8 | 19.723 | 14.642 | 10.677 | 15.014 | 638.969 | 16.764 | 1695.370 | 1.152 | 0.653 |
| phase_sq_int8_w4a16_ours | 19.839 | 13.459 | 10.516 | 14.605 | 550.600 | 6.676 | 971.461 | 2.011 | 1.140 |
| phase_int8_w4a16_ours | 19.807 | 13.727 | 10.618 | 14.718 | 550.797 | 6.636 | 970.143 | 2.013 | 1.141 |
| phase_fp8_w4a16_ours | 19.452 | 13.848 | 10.554 | 14.618 | 639.983 | 6.704 | 1062.348 | 1.839 | 1.042 |

## 真实质量诊断

既有WikiText test固定64窗口，非新盲测；保护值=mean+2SE+.005。不按NLL预先淘汰预定生成点。


| 固定配置 | 实测ΔNLL | 保护值 |
| --- | ---: | ---: |
| uniform_dense_bf16 | 0.00000 | 0.00000 |
| uniform_int8 | 0.01029 | 0.02046 |
| uniform_sq_int8 | 0.01294 | 0.02290 |
| uniform_fp8 | 0.00697 | 0.01568 |
| phase_sq_int8_w4a16_ours | 0.03176 | 0.04595 |
| phase_int8_w4a16_ours | 0.03518 | 0.04937 |
| phase_fp8_w4a16_ours | 0.03424 | 0.04841 |

## 固定分配

| 设置 | Prefill | Decode |
| --- | --- | --- |
| uniform_dense_bf16 | {'dense_bf16': 128} | {'dense_bf16': 128} |
| uniform_int8 | {'int8': 128} | {'int8': 128} |
| uniform_sq_int8 | {'sq_int8': 128} | {'sq_int8': 128} |
| uniform_fp8 | {'fp8': 128} | {'fp8': 128} |
| phase_sq_int8_w4a16_ours | {'sq_int8': 128} | {'w4a16_ours': 128} |
| phase_int8_w4a16_ours | {'int8': 128} | {'w4a16_ours': 128} |
| phase_fp8_w4a16_ours | {'fp8': 128} | {'w4a16_ours': 128} |

[全部配置与分项](results/sweep_all_results.csv) · [固定控制](results/sweep_results.csv) · [三轮速度与范围](results/sweep_formal_rounds.csv) · [数据与参数冻结](configs/sweep_design.json)

[原始代码与完整记录](../../../runs/038/README.md)
