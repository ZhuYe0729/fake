# 准备、评测和阶段切换开销（R1 AK6a / R3 LrT2 / R4 QboE / R5 gkP9）

统计叶子计算命令，排除driver/queue父进程。进程壁钟求和包含CPU、模型载入及GPU执行，不是实际GPU kernel占用时间，也不是并行后的总墙钟。失败耗时单列；正在运行的未完成命令不计入，不能将中间快照当作完整成本。

| Run | Stage | Success jobs | Success process-sum seconds | Failed jobs | Failed process-sum seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| 016 | allocation | 1 | 8.06 | 0 | 0.00 |
| 016 | calibration_speed | 23 | 824.04 | 0 | 0.00 |
| 016 | confirmation_nll | 37 | 1490.54 | 0 | 0.00 |
| 016 | export | 254 | 5285.47 | 0 | 0.00 |
| 016 | formal_speed | 114 | 4533.30 | 0 | 0.00 |
| 016 | full_generation | 21 | 2860.07 | 0 | 0.00 |
| 016 | full_mc | 4 | 3461.75 | 0 | 0.00 |
| 016 | local_error | 1 | 79.25 | 0 | 0.00 |
| 016 | quality_fit | 1 | 11.80 | 0 | 0.00 |
| 016 | training_nll | 50 | 2513.21 | 10 | 311.52 |
| 016 | validation_nll | 37 | 1330.32 | 1 | 12.93 |
| 017 | allocation | 1 | 6.75 | 0 | 0.00 |
| 017 | calibration_speed | 38 | 1384.62 | 0 | 0.00 |
| 017 | confirmation_nll | 52 | 2038.42 | 0 | 0.00 |
| 017 | export | 239 | 5193.70 | 0 | 0.00 |
| 017 | formal_or_anchor_speed | 1 | 59.27 | 0 | 0.00 |
| 017 | formal_speed | 69 | 3146.94 | 0 | 0.00 |
| 017 | full_generation | 51 | 17210.32 | 0 | 0.00 |
| 017 | full_mc | 13 | 10227.81 | 4 | 171.38 |
| 017 | hessian | 1 | 1142.91 | 0 | 0.00 |
| 017 | linear_profile | 1 | 41.43 | 0 | 0.00 |
| 017 | local_error | 1 | 105.17 | 0 | 0.00 |
| 017 | quality_fit | 1 | 11.03 | 0 | 0.00 |
| 017 | training_nll | 102 | 5178.46 | 0 | 0.00 |
| 017 | validation_nll | 52 | 1829.21 | 0 | 0.00 |
| 018 | allocation | 1 | 4.87 | 0 | 0.00 |
| 018 | calibration_speed | 31 | 1150.45 | 0 | 0.00 |
| 018 | confirmation_nll | 39 | 1569.89 | 0 | 0.00 |
| 018 | export | 206 | 5997.82 | 0 | 0.00 |
| 018 | formal_or_anchor_speed | 1 | 46.38 | 0 | 0.00 |
| 018 | formal_speed | 66 | 2759.92 | 0 | 0.00 |
| 018 | full_generation | 48 | 15935.59 | 0 | 0.00 |
| 018 | full_mc | 13 | 10223.34 | 4 | 196.32 |
| 018 | hessian | 1 | 1088.76 | 0 | 0.00 |
| 018 | linear_profile | 1 | 32.46 | 0 | 0.00 |
| 018 | local_error | 1 | 97.68 | 0 | 0.00 |
| 018 | quality_fit | 1 | 11.57 | 0 | 0.00 |
| 018 | training_nll | 102 | 5070.15 | 0 | 0.00 |
| 018 | validation_nll | 39 | 1453.22 | 0 | 0.00 |

## 精确权重缓存操作

以下是导出后的复制及SHA校验操作，已包含在其父队列壁钟内；与上述叶子进程统计分列，不能相加作为GPU时间或单次准备总墙钟。缓存仅复用完全相同的导出字节，所有primary方法采用相同规则。

| Run | Cached exports | Total GiB | Copy and verify process-sum seconds |
| --- | ---: | ---: | ---: |
| 017 | 21 | 202.29 | 583.46 |
| 018 | 20 | 138.63 | 400.34 |

[逐份权重缓存记录](weight_cache_operations.csv)

训练标签、validation和速度校准属于搜索/准备成本；confirmation和完整MC/生成是评测成本，分别报告。反复导出属于本次实验工程流程，既不能当作每请求成本，也不能隐去。

Qwen016继承每场景120标签、新增24，共144（108 train/36 holdout）；Llama017/018各继承72、新增48，共120（90 train/30 holdout）。缓存继承不是从零免费：Qwen原Hessian1765.64秒、两种SparseGPT各305秒；旧标签阶段壁钟及额外8bit标签成本详见run004，旧阶段相互重叠，不能直接相加宣称单次总时长。两个Llama的原SparseGPT状态/标签来源另行保留；当前重用前均通过数值门槛。

## Qwen016实际运行时成本

以下P+D三轮中位数。初始化含引擎准备，arena捕获为其中子项，不能相加重复计费；NVML是两场景同进程的采样峰值。host cache与GPU arena均保留，恢复计入cycle；E2E延续原计时边界。

| Policy | Init s | Arena capture ms | NVML peak GiB | Host weights GiB | GPU arena GiB | Restore ms | Cycle ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform_dense_bf16 | 11.76 | 710.33 | 25.29 | 0.00 | 0.00 | 0.16 | 1991.80 |
| uniform_int8 | 8.76 | 732.08 | 20.01 | 0.00 | 0.00 | 0.18 | 1227.35 |
| uniform_sq_int8 | 8.86 | 781.40 | 20.02 | 0.00 | 0.00 | 0.19 | 1232.89 |
| hardware_minimum_prefill_decode_phase_pair | 11.52 | 746.60 | 18.28 | 3.80 | 1.90 | 49.65 | 1112.95 |
| daso8_exact_prefill_decode_q0.04 | 16.60 | 980.64 | 18.86 | 8.70 | 5.06 | 111.09 | 1082.18 |

DASO-8高预算点的E2E收益大于包含恢复的cycle收益，host/arena占用也高于共同硬件端点；不将其解释为免费阶段切换或所有负载均有同等收益。

[叶子命令/哈希](preparation_processes.csv) · [按阶段汇总](preparation_summary.csv) · [逐轮运行时原始汇总](qwen016_runtime_overhead.csv) · [旧准备成本](../runs/004/results/preparation_costs.json)
