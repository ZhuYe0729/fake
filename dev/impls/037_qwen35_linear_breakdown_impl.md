## 2026-06-05 - Qwen3.5 单层 kernel breakdown debug
- 开发目的：拆解 Qwen3.5-9B `mlp.down_proj` 在 normal_01 batch_size=1 场景下的 sparse_bf16 与 dense_nvfp4/marlin 路径耗时。
- 修改内容：新增 debug 脚本，测量真实权重单层的 build/conversion、first forward、steady prefill/decode，并输出 JSON/CSV/README。
- 影响文件：`artifacts/debug/001_qwen35_linear_breakdown/`、`dev/plans/037_qwen35_linear_breakdown_plan.md`、本实现记录。
- 测试结果：`sparse_bf16` steady runtime 为 12.2057ms；显式 `dense_nvfp4` prefill + `marlin_nvfp4` decode 为 10.3659ms；lazy wrapper 为 10.2629ms。`sparse_bf16` build/pack 单层约 18.4s，NVFP4 canonical+CUTLASS+Marlin 显式准备约 190ms。
- 后续注意：该脚本为 debug 用途，不接入主 benchmark 或 predictor policy；build/materialization 属于离线或首次调用成本，解释 E2E 时需要和 steady runtime 分开看。

## 2026-06-06 - 补测多个 W4A4/W4A16 代表 linear
- 开发目的：额外测试 Qwen3.5-9B 中多个曾被策略选择为 `dense_nvfp4/marlin_nvfp4` 的 linear，观察单层 breakdown 是否一致。
- 修改内容：debug 脚本支持 `--layers` 和 `--default-w4a4-w4a16-layers`，一次加载模型后连续测多个层；新增 aggregate JSON/CSV/README 和每层子目录 README。
- 影响文件：`artifacts/debug/001_qwen35_linear_breakdown/scripts/qwen35_linear_breakdown.py`、`artifacts/debug/001_qwen35_linear_breakdown/results/aggregate_breakdown.*`、各层 `results/<layer>/` 子目录。
- 测试结果：6 个代表层全部完成；steady runtime 下显式 `dense_nvfp4` prefill + `marlin_nvfp4` decode 均快于 `sparse_bf16`，差距约 1.92ms 到 3.47ms/层。主要优势来自 decode：marlin decode step 约 0.043-0.047ms，而 sparse_bf16 decode step 约 0.124-0.218ms。
- 后续注意：本次多层测试使用 `warmup=3,iters=10`，用于 debug 趋势判断；与前一次单层 `warmup=5,iters=20` 数值不完全逐点可比。

## 2026-06-06 - 准备 full-model 内部 linear trace
- 开发目的：回应 standalone 单层测试不等同于真实端到端路径的问题，直接在完整 Qwen3.5-9B forward 中 hook 指定 linear，测真实 prefill/decode 调用耗时。
- 修改内容：新增 `qwen35_full_model_linear_trace.py`，支持对 `sparse_bf16` 和 `predictor_hybrid` 两种 full model 替换路径注册 forward hook，输出 per-call trace、per-layer summary 和 full E2E 时间。
- 影响文件：`artifacts/debug/001_qwen35_linear_breakdown/scripts/qwen35_full_model_linear_trace.py`、后续 `full_model_trace/` 结果目录。
- 后续注意：hook 内会同步 CUDA 来获得单层时间，因此该 debug run 的模型总时延会被 instrumentation 扰动；这里重点看真实模块调用的相对 breakdown，而不是替代无 hook E2E benchmark。

## 2026-06-06 - 完成 full-model trace 与无 hook 复核
- 开发目的：验证真实 full model forward 中目标 linear 的耗时，并复核同场景无 hook E2E 是否仍表现为 `sparse_bf16` 更快。
- 修改内容：运行 `qwen35_full_model_linear_trace.py`，生成 `full_model_trace/linear_trace.*`、`linear_trace_summary.csv`、`FULL_MODEL_TRACE.md`；额外用无 hook benchmark 复测 `sparse_bf16` 与 `predictor_hybrid`。
- 影响文件：`artifacts/debug/001_qwen35_linear_breakdown/full_model_trace/`、`FULL_MODEL_TRACE.md`、本实现记录。
- 测试结果：hook trace 中 6 个目标层合计 `predictor_hybrid` 比 `sparse_bf16` 少约 96.13ms prefill 和 71.35ms decode；无 hook E2E 下 `sparse_bf16=5034.61ms`，`predictor_hybrid=4400.60ms`。
- 后续注意：在严格对齐 `batch_size=1,input_tokens=16384,output_tokens=32,warmup=0` 的当前环境下，没有复现 `sparse_bf16` 比 predictor hybrid 更快；此前结论需要回查是否混入了不同 batch/input 场景、manual summary 口径或 benchmark 脚本差异。

## 2026-06-06 - 直接 E2E warmup=3 复测
- 开发目的：按普通 benchmark 风格直接复测完整模型 E2E，确认 hook 之外、warmup 开启时两种方法的真实性能。
- 修改内容：运行 `bench_qwen3_5_swh_e2e.py`，只测 `sparse_bf16` 与 `predictor_hybrid`，场景为 `batch_size=1,input_tokens=16384,output_tokens=32,warmup_iters=3`。
- 影响文件：`artifacts/debug/001_qwen35_linear_breakdown/full_model_trace/no_hook_e2e_sparse_vs_predictor_warmup3.csv`、`FULL_MODEL_TRACE.md`、本实现记录。
- 测试结果：`sparse_bf16=4611.55ms`，`predictor_hybrid=4204.78ms`；predictor_hybrid 快 406.78ms。
- 后续注意：该复测进一步说明 bs=1 normal 场景下此前 `sparse_bf16` 更快的结论不成立，需要回查是否使用了 Qwen manual 的 bs=32/in512 场景或旧结果。
