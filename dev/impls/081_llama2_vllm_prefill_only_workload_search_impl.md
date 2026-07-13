## 2026-07-09 - 完成 prefill-only 场景筛选分析

- 开发目的：为 Llama2-7B + vLLM 寻找适合展示层异构压缩优势的 prefill-only `(batch_size,input_seq)` 配置。
- 修改内容：新增 focused analyzer，复用 P024 质量预算、Llama2 fused Linear 形状、CUTLASS `KernelLatencyPredictor` 和已有 promising retest 结果；生成纯 prefill 预测候选表、已有 out=1 实测汇总、focused retest 场景和运行脚本；补充推荐结论。
- 影响文件：`dev/plans/081_llama2_vllm_prefill_only_workload_search_plan.md`，`artifacts/exports/vllm/llama2_7b_018/prefill_only_workload_search/scripts/analyze_prefill_only_workloads.py`，`artifacts/exports/vllm/llama2_7b_018/prefill_only_workload_search/summary/`。
- 后续注意：现有数据和纯 prefill 预测都不支持“有意义的大 prefill-only 场景中，P024 质量约束 hetero 明显优于所有统一压缩方法”。已测最稳妥 prefill-only proxy 是 `b8_in512_out1` 的 `max_speed_hetero`，但仅比最佳统一 `sparse_nvfp4` 快 `1.020x`，精度 `acc_norm` 从 `0.4514` 降到 `0.4087`。当前磁盘使用率 99%，未直接导出新的 focused hetero checkpoints。

## 2026-07-09 - 实测不看精度的 prefill-only max-speed 场景

- 开发目的：验证不看精度时，prefill-only 场景下 max-speed hetero 是否能相对 best single 明显提速。
- 修改内容：使用空闲 RTX 5090 复测 `b8_in2048_out1` 和 `b1_in512_out1`；统一方法使用 `dense_bf16/dense_nvfp4/sparse_bf16/sparse_nvfp4/marlin_nvfp4`，hetero 复用已有 `maxspeed_004_f2600ffcfc` 和 `maxspeed_005_4746310a30` checkpoint；每个配置 1 warmup + 5 iters。
- 影响文件：`artifacts/exports/vllm/llama2_7b_018/prefill_only_workload_search/b8_in2048_uniform_vllm_env/`，`b8_in2048_maxspeed_hetero_vllm/`，`b1_in512_uniform_vllm_env/`，`b1_in512_maxspeed_hetero_vllm/`，以及 `summary/actual_prefill_only_speed_test.*`。
- 后续注意：`b8_in2048_out1` 实测 only `1.016x` over best single `sparse_nvfp4`，不算明显；`b1_in512_out1` 实测 `1.222x` over best single `sparse_bf16`，可以作为 speed-only prefill-only 展示点，但它是较小 workload，容易被 vLLM 固定开销影响。

## 2026-07-09 - 整理 prefill-only speed policy 的真实 checkpoint 精度

- 开发目的：对比两个 speed-only prefill 场景所用 max-speed hetero policy、未压缩 dense bf16、全局统一压缩方法的 full ARC-Challenge 精度。
- 修改内容：复用已有 vLLM/lm-eval full ARC-Challenge 0-shot 结果，合并 `dense_bf16/dense_nvfp4/sparse_bf16/sparse_nvfp4/marlin_nvfp4` 和 `maxspeed_004_f2600ffcfc/maxspeed_005_4746310a30`；检查 `export_summary.json` 确认压缩结果均为导出的 vLLM checkpoint，不是运行时模块替换。
- 影响文件：`artifacts/exports/vllm/llama2_7b_018/prefill_only_workload_search/summary/actual_prefill_only_quality_comparison.csv`，`actual_prefill_only_quality_comparison.md`。
- 后续注意：`b8_in2048_out1` 对应 hetero policy `acc_norm=0.4087`，低于 uniform `dense_nvfp4/marlin_nvfp4` 但高于 `sparse_bf16/sparse_nvfp4`；`b1_in512_out1` 对应 hetero policy `acc_norm=0.2884`，低于 `sparse_bf16`，仅高于 `sparse_nvfp4`。
