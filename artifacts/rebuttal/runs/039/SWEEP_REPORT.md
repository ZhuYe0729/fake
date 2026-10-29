# llama2_7b_chat：统一8bit-prefill/W4A16-decode控制

固定两种既有全局阶段map，与已有SQ-prefill/W4-decode及uniform点比较；所有128模块的prefill分别为plain INT8或FP8，decode均W4A16。不是DASO求解器新选点，也未改精度模型。

本轮无代理预算/权重调参；NLL只作质量诊断，无strictε或优化器获胜声明。全部固定控制保留，原Prefill-only/原主表不改，无新增MC。

速度统一GPU6/NUMA1，B8/S2048/O64，原KV10GiB、native decode Graph；三轮3warmup+5timed、BF16首尾锚点、同轮INT8。E2E计入prefill→decode切换，不计下一批恢复。

完整生成保留原PMPD/BOS、batch4、greedy最多256新token/EOS；1000/1500/333全样本。相同map/源码/输入协议的018结果按SHA复用，其他新测。Avg是ROUGE-L%/BLEU三项算术摘要，不是统一准确率。

| 设置 | CNN/DM | DialogSum | IWSLT | Gen Avg | TTFT ms | TPOT ms | E2E ms | ×BF16 | ×INT8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform_dense_bf16 | 23.604 | 21.608 | 19.441 | 21.551 | 1035.244 | 15.638 | 2020.611 | 1.000 | 0.639 |
| uniform_int8 | 23.663 | 21.746 | 16.873 | 20.761 | 528.706 | 12.108 | 1291.685 | 1.564 | 1.000 |
| uniform_sq_int8 | 23.555 | 21.318 | 18.203 | 21.026 | 528.504 | 12.120 | 1292.093 | 1.564 | 1.000 |
| uniform_fp8 | 23.615 | 21.797 | 19.047 | 21.486 | 603.605 | 18.797 | 1788.323 | 1.130 | 0.722 |
| phase_sq_int8_w4a16_ours | 23.898 | 21.655 | 18.469 | 21.341 | 529.938 | 9.816 | 1148.438 | 1.759 | 1.125 |
| phase_int8_w4a16_ours | 23.893 | 21.286 | 17.546 | 20.909 | 529.251 | 9.847 | 1149.548 | 1.758 | 1.124 |
| phase_fp8_w4a16_ours | 23.770 | 22.118 | 18.589 | 21.492 | 605.561 | 9.853 | 1226.177 | 1.648 | 1.053 |

## 真实质量诊断

既有WikiText test固定64窗口，非新盲测；保护值=mean+2SE+.005。不按NLL预先淘汰预定生成点。


| 固定配置 | 实测ΔNLL | 保护值 |
| --- | ---: | ---: |
| uniform_dense_bf16 | 0.00000 | 0.00000 |
| uniform_int8 | 0.00434 | 0.01718 |
| uniform_sq_int8 | 0.02201 | 0.03592 |
| uniform_fp8 | 0.00591 | 0.01553 |
| phase_sq_int8_w4a16_ours | 0.03408 | 0.05193 |
| phase_int8_w4a16_ours | 0.01373 | 0.03143 |
| phase_fp8_w4a16_ours | 0.01675 | 0.03164 |

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
