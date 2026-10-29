# Qwen3完整空间：prefill权重尺度补充

固定016八候选Qwen代理，BF16零锚定；144模块独立选择64个阶段组合。λ只调整prefill质量代价，不禁止格式。先最小延迟、再同延迟最小预测质量损失。

本补充在已看到032部分预算曲线后提出，不冒充原预定六点或新盲测。原代理gate_up稠密FP4的prefill总成本仅约.000858，全W4 decode约.010915；λ1.5/2不足以明显改变阶段分配。据CPU代价尺度与结构变化固定λ32/128、ε=.03，在读取本补充GPU输出前冻结，全部点保留。λ≠1得分不是重新校准后的真实ΔNLL。

本补充倍率较大，不称为微小调整。NLL/完整生成/MC的BF16和INT8对照按协议及来源从032复用，速度为本轮统一重测；完整生成Avg为两个ROUGE-L百分分数与BLEU的算术摘要，MC Avg为四项准确率等权均值。

对decode固定全W4的子空间，q_decode为常数，λ·q_prefill+q_decode≤ε可等价改写为原得分≤ε/λ+(1−1/λ)q_decode。本轮λ=1.5/2的选点均在该子空间，因此需先与收紧预算区分，不把加权本身宣称为超越预算曲线的新优势；完整搜索仍允许其他decode配置。

## 完整生成与三轮速度

| Label | λ | ε | Prefill sparse FP4 | CNN/DM | DialogSum | IWSLT | Avg | E2E ms | ×INT8 | ×BF16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| w32_e0.03 | 32.0 | 0.03 | 0 | 22.586 | 27.359 | 39.909 | 29.951 | 988.20 | 1.240 | 2.026 |
| w128_e0.03 | 128.0 | 0.03 | 0 | 22.481 | 27.340 | 39.882 | 29.901 | 1029.75 | 1.190 | 1.945 |
| uniform_dense_bf16 | — | — | 0 | 22.824 | 26.957 | 40.132 | 29.971 | 1998.47 | 0.613 | 1.002 |
| uniform_int8 | — | — | 0 | 22.747 | 26.822 | 39.586 | 29.718 | 1225.51 | 1.000 | 1.634 |

## 对应P+D配置的完整prefill MC精度

MC使用各P+D配置自身的prefill，没有另行求解prefill-only策略。原run016主表的两个场景分别选点，两者需要区分。

| Label | ARC-C | ARC-E | Wino | MMLU | Avg acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| w32_e0.03 | 56.997 | 81.566 | 68.272 | 72.226 | 69.765 |
| w128_e0.03 | 55.717 | 81.987 | 67.956 | 71.856 | 69.379 |
| uniform_dense_bf16 | 56.911 | 83.460 | 68.035 | 72.874 | 70.320 |
| uniform_int8 | 56.911 | 82.744 | 67.877 | 72.333 | 69.966 |

MC只做prompt likelihood，保持原prefill权重/内核，关闭不使用的decode搬运，结束断言没有调度/恢复。四任务完整、0-shot、batch4、eager/3GiB KV；基线来自032同协议结果。生成同map/源码/输入协议复用，来源单列于full_generation.csv。

NLL为既有WikiText test固定100窗口，未称为全新盲测/独立确认；全部点评测，不依据NLL或最终任务分数挑λ/重选赢家。加权得分、原预测与实际NLL分别列于sweep_results.csv；保护值仍mean+2SE+.005，只作预定名义预算的诊断。

速度GPU0/NUMA0，B8/S2048/O64，6GiB KV/native decode Graph；三轮3warmup+5timed，首尾BF16。E2E含prefill→decode实际切换，不包含下一批恢复；恢复和cycle另存。MC表没有把P+D TTFT冒充额外测过的prefill-only速度。

本轮未迁卡：第三轮两个计时进程之间曾等待GPU0租约，等待时间不计入E2E；资源备选入口检测到原测已继续后退出，没有停止控制器或其他进程。记录见results/resource_wait.json。

[全部分项](results/sweep_results.csv) · [INT8相对比较](results/relative_to_int8.csv) · [三轮速度](results/formal_campaigns.csv) · [完整生成](results/full_generation.csv) · [MC](results/full_mc.csv) · [NLL](results/benchmark_nll.csv) · [配置分布](results/policy_counts.json)
