# Llama2-7B vLLM Broad Grid Benchmark Plan

## Summary

- 在 `artifacts/exports/vllm/llama2_7b_018/broad_grid_vllm/` 下进行更广泛的 vLLM 速度网格测试。
- 网格为 batch 13 档、input_seq 9 档、output_seq 4 档，共 468 个配置。
- 方法全测：`dense_bf16`、`dense_nvfp4`、`sparse_bf16`、`sparse_nvfp4`、`marlin_nvfp4`、`hetero`。
- 每个可运行配置跑 `1 warmup + 3 timed iters`，主结果表填 median latency ms；失败配置保留行并填失败状态。

## Key Changes

- 新增 broad-grid 单方法 benchmark 脚本，支持 OOM/初始化失败捕获、长上下文 `hf_overrides`、以及 `_CUDA_COMPAT_STATUS` 清理。
- 新增 broad-grid 并行 launcher，每张 GPU 只跑一个 method 进程，不做 tensor parallel。
- 新增汇总脚本，输出 long summary、iterations、latency 大表、speedup 大表和 Markdown 报告。
- hetero 策略按 selected-8 外推：小 batch/小 M 使用 A/B，其他使用 C。

## Test Plan

- 先跑 tiny smoke grid，验证 6 个方法可调度、结果可合并、表格格式正确。
- 正式运行使用 6 张 GPU 并行，默认 `1 warmup + 3 iter`。
- 汇总检查大表行数为 468，方法列为 latency ms 或 `PRECHECK_OOM` / `INIT_OOM` / `OOM` / `ERROR`。

## Assumptions

- 只测试速度，不重新测试精度。
- 使用已有 vLLM 导出 checkpoint。
- 对明显超过单卡可承受规模的配置做 prompt-token 预检查并标记 `PRECHECK_OOM`，避免主机内存被超大 prompt list 耗尽。
