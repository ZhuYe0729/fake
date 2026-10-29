# Qwen新点真实chunked/mixed调度（R2 Exi5 / R1 AK6a / R5 gkP9）

040固定九map、006原两个代表case，实际vLLM V1 chunked prefill预算1024、同forward的prefill/decode逐行路由。所有请求t0进入；这是closed-loop burst、dual-resident eager变体，不能称HTTP/open-loop生产服务或原单表示主表性能。

固定GPU6/NUMA1、KV6GiB/no prefix cache。三轮每轮2warmup+5timed，首尾BF16；每轮BF16两端中位数算术平均为分母，再取三轮加速算术均值。BF16展示此锚点均值，其他方法展示各轮中位数均值。p50/p95/p99来自15个timed burst的120条请求观测，是描述性burst分位数。

质量为019原long_quality中64个已查看窗口，在各case实际prompt/输出长度上paired teacher forcing；不是新盲测或下游生成任务分数。保护值=mean+2SE+.005（BF16为0），不把旧名义q当此场景质量保证。BF16 routed/no-op逐窗口NLL通过；真实mixed forward及阶段总token数逐burst核验，跨格式模块精确gather/apply/scatter审计，计时外核验GPU/CPU阶段权重一致。

| Case | 方法 | E2E ×BF16 | ×INT8 三轮范围 | ΔNLL±2SE | 保护值 | p95 TTFT ms | p95 TPOT ms | GPU adapter字段GiB | Torch峰值GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk1024_b8_s2048_o64 | BF16 | 1.000 | 0.878–1.203 | 0.00000±0.00000 | 0.00000 | 1414.1 | 30.41 | 12.94 | 21.38 |
| chunk1024_b8_s2048_o64 | INT8 W8A8 | 0.940 | 1.000–1.000 | -0.02149±0.01412 | -0.00237 | 652.3 | 34.80 | 6.47 | 14.99 |
| chunk1024_b8_s2048_o64 | SmoothQuant INT8 | 0.963 | 0.783–1.280 | -0.02845±0.01212 | -0.01132 | 645.2 | 34.01 | 6.48 | 14.99 |
| chunk1024_b8_s2048_o64 | FP8 W8A8 | 1.037 | 1.020–1.171 | 0.00605±0.00516 | 0.01621 | 823.5 | 30.44 | 6.48 | 14.93 |
| chunk1024_b8_s2048_o64 | Uniform SQ-INT8 prefill / W4A16 decode | 1.290 | 1.140–1.575 | 0.02104±0.01358 | 0.03962 | 787.5 | 24.40 | 10.11 | 18.65 |
| chunk1024_b8_s2048_o64 | HAWQ-FP4 PTQ r=.80 | 0.925 | 0.950–1.020 | -0.00285±0.00960 | 0.01175 | 1024.6 | 35.08 | 9.12 | 17.73 |
| chunk1024_b8_s2048_o64 | Common hardware-minimum control | 1.015 | 0.888–1.391 | -0.02500±0.01488 | -0.00512 | 703.3 | 33.28 | 6.89 | 15.37 |
| chunk1024_b8_s2048_o64 | DASO exploratory w1 q=.01 | 1.348 | 1.292–1.679 | 0.00506±0.01817 | 0.02823 | 736.0 | 24.35 | 8.59 | 17.10 |
| chunk1024_b8_s2048_o64 | DASO exploratory w32 q=.03 | 1.225 | 1.050–1.560 | 0.01504±0.01429 | 0.03432 | 912.4 | 26.13 | 8.76 | 17.31 |
| chunk1024_b8_heterogeneous_o256 | BF16 | 1.000 | 1.478–1.497 | 0.00000±0.00000 | 0.00000 | 879.0 | 16.27 | 12.94 | 21.38 |
| chunk1024_b8_heterogeneous_o256 | INT8 W8A8 | 0.673 | 1.000–1.000 | -0.01806±0.01059 | -0.00247 | 395.9 | 24.57 | 6.47 | 14.99 |
| chunk1024_b8_heterogeneous_o256 | SmoothQuant INT8 | 0.674 | 0.997–1.009 | -0.02071±0.00832 | -0.00740 | 392.1 | 24.36 | 6.48 | 14.99 |
| chunk1024_b8_heterogeneous_o256 | FP8 W8A8 | 0.783 | 1.146–1.175 | 0.00590±0.00382 | 0.01472 | 506.8 | 20.92 | 6.48 | 14.93 |
| chunk1024_b8_heterogeneous_o256 | Uniform SQ-INT8 prefill / W4A16 decode | 1.205 | 1.784–1.795 | 0.02727±0.00946 | 0.04173 | 478.3 | 13.56 | 10.11 | 18.65 |
| chunk1024_b8_heterogeneous_o256 | HAWQ-FP4 PTQ r=.80 | 0.776 | 1.129–1.184 | 0.00584±0.00637 | 0.01721 | 610.7 | 21.11 | 9.12 | 17.73 |
| chunk1024_b8_heterogeneous_o256 | Common hardware-minimum control | 0.732 | 1.080–1.098 | -0.00561±0.01161 | 0.01100 | 398.7 | 22.41 | 6.89 | 15.37 |
| chunk1024_b8_heterogeneous_o256 | DASO exploratory w1 q=.01 | 1.013 | 1.492–1.526 | 0.02608±0.01114 | 0.04223 | 468.0 | 16.17 | 8.59 | 17.10 |
| chunk1024_b8_heterogeneous_o256 | DASO exploratory w32 q=.03 | 1.210 | 1.778–1.827 | 0.02122±0.01119 | 0.03741 | 472.3 | 17.57 | 8.76 | 17.31 |

GPU/host adapter字段按实际storage去重，是该字段子集；Torch峰值另含KV、workspace和其余模型张量，不能将二者相加。dual-resident为消除阶段释放约束付出显存，主表arena恢复成本不移植为这里的异步收益。完整CSV另列初始化、host字段、reserved峰值、非Torch显存诊断和p50/p99。两case/三轮不构成稳定生产SLA或所有负载全局最优保证。

[完整表](mixed_results.csv) · [三轮](mixed_rounds.csv) · [首尾BF16](bf16_brackets.csv) · [实际mixed计数](routing_counts.csv) · [设计/代码/原始captures](../../../runs/047/README.md) · [来源](provenance.json)
