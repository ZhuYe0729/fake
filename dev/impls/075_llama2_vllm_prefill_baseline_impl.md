## 2026-07-07 - vLLM prefill baseline setup
- 开发目的：把 Llama2-7B 原始 BF16 和 uniform 压缩模型的 vLLM prefill 速度测试固定为 `artifacts/exports/vllm/llama2_7b_018` 下的 baseline。
- 修改内容：新增计划记录；新增 baseline README；新增 `benchmark_prefill_vllm.py`，使用 batch size 16、prompt length 1024、`max_tokens=1`、关闭 prefix cache、统一 eager 模式测试 dense BF16/dense NVFP4/sparse BF16/sparse NVFP4。
- 影响文件：`dev/plans/075_llama2_vllm_prefill_baseline_plan.md`，`dev/impls/075_llama2_vllm_prefill_baseline_impl.md`，`artifacts/exports/vllm/llama2_7b_018/README.md`，`artifacts/exports/vllm/llama2_7b_018/scripts/benchmark_prefill_vllm.py`。
- 后续注意：该 vLLM generate 口径是 `prefill_plus_1_decode`，不是严格 `output_tokens=0`；custom quant backend 当前需要 `enforce_eager=True`。

## 2026-07-07 - Quality baseline and vLLM worker timing fix
- 开发目的：复用已有完整精度结果，并修复本机 vLLM V1 worker/exclusive GPU 模式下的计时问题。
- 修改内容：生成 `quality/uniform_full_arc_c_quality.csv` 和说明文件，来源为 018 full ARC-C 1172 样本结果；移除 benchmark 脚本主进程中的 `torch.cuda.synchronize()`，避免 vLLM worker 进程持有 GPU 时主进程同步触发 device busy。
- 影响文件：`artifacts/exports/vllm/llama2_7b_018/quality/`，`artifacts/exports/vllm/llama2_7b_018/scripts/benchmark_prefill_vllm.py`，`dev/impls/075_llama2_vllm_prefill_baseline_impl.md`。
- 后续注意：计时围绕 `llm.generate()` wall time；vLLM 返回时请求已完成。

## 2026-07-07 - Full four-model vLLM baseline results
- 开发目的：完成 dense BF16、dense NVFP4、sparse BF16、sparse NVFP4 的固定 vLLM prefill baseline。
- 修改内容：运行完整 benchmark，warmup 2 次、计时 10 次；生成逐 iteration CSV、每方法 summary JSON、总 speed CSV、speed+quality 汇总 CSV/Markdown；README 增加当前 baseline 表。
- 影响文件：`artifacts/exports/vllm/llama2_7b_018/benchmarks/prefill_vllm/`，`artifacts/exports/vllm/llama2_7b_018/summary/`，`artifacts/exports/vllm/llama2_7b_018/README.md`，`dev/impls/075_llama2_vllm_prefill_baseline_impl.md`。
- 后续注意：当前固定结果为 dense BF16 1051.619 ms、dense NVFP4 563.644 ms、sparse BF16 631.798 ms、sparse NVFP4 504.653 ms；对应 speedup 为 1.000x、1.866x、1.664x、2.084x。
