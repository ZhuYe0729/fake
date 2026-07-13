## 2026-07-07 - 初始化 broad-grid benchmark

- 开发目的：实现 Llama2-7B vLLM 广泛 batch/input/output 网格速度测试。
- 修改内容：创建计划文件和实现记录，准备新增 broad-grid benchmark、并行和汇总脚本。
- 影响文件：`dev/plans/077_llama2_vllm_broad_grid_benchmark_plan.md`，`dev/impls/077_llama2_vllm_broad_grid_benchmark_impl.md`。
- 后续注意：正式运行需清理 `_CUDA_COMPAT_STATUS` 并保证每张 GPU 只运行一个 method 进程。

## 2026-07-07 - 完成 broad-grid vLLM 测试

- 开发目的：按 batch/input/output 广泛网格测试 Llama2-7B 各 vLLM 导出方法的速度结果，并生成 baseline 表格。
- 修改内容：新增 broad-grid 单方法 benchmark、按 GPU 并行调度、汇总脚本和一键运行脚本；完成 6 个方法、每方法 468 个配置的测试与汇总；修正平均 speedup 只统计 dense_bf16 可比 OK 行。
- 影响文件：`artifacts/exports/vllm/llama2_7b_018/broad_grid_vllm/scripts/`，`artifacts/exports/vllm/llama2_7b_018/broad_grid_vllm/results/`，`artifacts/exports/vllm/llama2_7b_018/broad_grid_vllm/summary/`。
- 后续注意：`summary_long.csv` 保留 PRECHECK/INIT_ERROR 等不可运行状态；`b8,input16384,out128` 等 dense_bf16 边界点耗时很长，后续若做快速回归可单独缩小网格或降低 iters。

## 2026-07-07 - 筛选高潜力场景并补充建模分析

- 开发目的：从 broad-grid 结果中找出存在明显加速空间的合适场景，并用 kernel latency modeling 分析 single 方法和最佳层异构混合方法的 Linear 部分速度。
- 修改内容：新增 `analyze_promising_scenarios_modeling.py`，按 vLLM 实测 speedup 筛选长上下文生成、prefill-only 和中等输出场景；用 fused Llama2 Linear shape 调用 `KernelLatencyPredictor`，输出 single 方法预测、best mixed 预测和逐 Linear 选择细节。
- 影响文件：`artifacts/exports/vllm/llama2_7b_018/broad_grid_vllm/scripts/analyze_promising_scenarios_modeling.py`，`artifacts/exports/vllm/llama2_7b_018/broad_grid_vllm/summary/promising_scenarios_modeling.md`，`artifacts/exports/vllm/llama2_7b_018/broad_grid_vllm/summary/promising_scenarios_modeling.csv`，`artifacts/exports/vllm/llama2_7b_018/broad_grid_vllm/summary/promising_scenarios_modeling_details.csv`。
- 后续注意：当前 best mixed 是速度上界分析，不含精度约束；single sparse 方法在 modeling 约束下可能因 decode `M` 不满足 kernel 约束而显示为空，具体原因保存在 CSV 的 `pred_*_status` 列。
