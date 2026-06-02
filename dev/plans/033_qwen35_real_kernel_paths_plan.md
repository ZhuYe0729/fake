# 033 Qwen3.5 Real Kernel Paths Plan

## Summary
为 Qwen3.5 文本模型补齐 5 条真实端到端速度路径：`dense`、`dense_nvfp4`、`sparse_bf16`、`sparse_nvfp4`、`marlin_nvfp4`。压缩路径均先 prepare packed checkpoint，benchmark 只加载压缩后权重，不在线 pack。

## Key Changes
- 新增 Qwen3.5 packed kernel checkpoint helper，覆盖 CUTLASS dense NVFP4、sparse BF16、sparse NVFP4 和 Marlin W4A16。
- 新增 `scripts/prepare_qwen3_5_kernel_checkpoint.py`，统一准备 Qwen3.5 真实 kernel checkpoint。
- 扩展 `scripts/bench_qwen3_5_speed.py --method` 支持 5 种真实路径，并按 method 默认查找对应 checkpoint。

## Test Plan
- `python -m py_compile` 覆盖新增 helper、prepare 脚本和 benchmark。
- `bash`/CLI help 检查 prepare/benchmark 参数。
- GPU 环境下分别 prepare 4 个 compressed checkpoint，再跑 5 个 method 的小规模 speed smoke。

## Assumptions
- 首版只覆盖 Qwen3.5 text-only `AutoModelForCausalLM` benchmark。
- sparse 路径在 prepare 阶段从 dense BF16 权重 prune+pack。
- runtime 加载 packed checkpoint，不重新量化/pack。
