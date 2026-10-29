# Validation窗口数量敏感性：Llama-2 P+D（R1 AK6a / R5 gkP9）

14个预先固定的主表内DASO可选incumbent，32窗口选择先严格复现原结果。新增32个不重叠validation窗口，仍按mean+2SE+.005选择，BF16保护值0。两种选择在同64个新确认窗口比较，确认窗口与018原窗口不重叠；选择在打开新确认之前冻结。原主表及原完整候选池结果保持不变。

| Val windows | ε | Policy | Val guard | Fresh test ΔNLL ±2SE | E2E × | CNN/DM | DialogSum | IWSLT | Mean exceeds ε |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 0.01 | uniform_dense_bf16 | 0.00000 | 0.00000 ± 0.00000 | 1.000 | 23.60 | 21.61 | 19.44 | False |
| 32 | 0.04 | uniform_int8 | 0.02555 | -0.00146 ± 0.00903 | 1.533 | 23.66 | 21.75 | 16.87 | False |
| 64 | 0.01 | uniform_dense_bf16 | 0.00000 | 0.00000 ± 0.00000 | 1.000 | 23.60 | 21.61 | 19.44 | False |
| 64 | 0.04 | hardware_minimum_prefill_decode_phase_pair | 0.03468 | 0.01219 ± 0.01254 | 1.789 | 23.67 | 21.57 | 17.40 | False |

速度和三项完整生成复用已审计018相同有效map的结果，不是新增独立下游测试；本实验只新增NLL窗口。这是有界incumbent池的样本量跟进研究，不是重新优化完整候选空间，不能替换原主表或据此宣称普遍最优。

[完整CSV](validation_size.csv) · [冻结设计](../../../runs/026/configs/design.json) · [新窗口](../../../runs/026/configs/frozen_design.json)

[独立实验代码与数据](../../../runs/026/README.md)
