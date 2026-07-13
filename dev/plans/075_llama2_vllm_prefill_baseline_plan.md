# Llama2-7B vLLM Prefill Baseline Plan

## Summary
- 在 `artifacts/exports/vllm/llama2_7b_018/` 下建立长期 baseline，测试原始 dense BF16 和三个已导出的 uniform 压缩模型的 vLLM prefill 场景速度。
- 精度结果直接复用 018 full ARC-Challenge 结果，样本数为 1172，不使用 limit-128 结果。
- vLLM benchmark 使用固定近似口径：batch size 16、prompt length 1024、`max_tokens=1`、`detokenize=False`、关闭 prefix cache。

## Key Changes
- 新增 baseline benchmark 脚本到 `artifacts/exports/vllm/llama2_7b_018/scripts/benchmark_prefill_vllm.py`。
- 新增 quality baseline 汇总，复制 uniform dense BF16/dense NVFP4/sparse BF16/sparse NVFP4 的 full ARC-C/NLL 指标。
- 输出速度结果到 baseline 目录内的 `benchmarks/` 和 `summary/`，用于后续固定引用。

## Test Plan
- 使用 `vllm` conda 环境和本机 RTX 5090 运行四个模型。
- 每个模型保存逐 iteration 延迟、均值、中位数、标准差、tokens/s 和相对 dense BF16 speedup。
- 对 `uniform_dense_nvfp4` 至少保留一次 vLLM eager generation smoke 通过记录。

## Assumptions
- vLLM custom quant backend 当前需要 `enforce_eager=True`，因此四个模型统一使用 eager 口径。
- vLLM generate API 不直接使用 `output_tokens=0`，本 baseline 记录为 `prefill_plus_1_decode`。
