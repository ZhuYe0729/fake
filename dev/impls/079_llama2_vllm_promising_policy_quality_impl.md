## 2026-07-07 - 完成 optimized policy full ARC-C 精度评测

- 开发目的：为 promising scenario 中重新求解的 8 个 unique optimized hetero policy 补充严格导出 checkpoint 后的 vLLM 精度结果，并更新最终对比宽表。
- 修改内容：新增 single-policy 和多 GPU parallel 的 vLLM + lm-eval ARC-Challenge 评测脚本；完成 8 个 policy 的 full ARC-Challenge 0-shot 评测；扩展 summary 脚本输出 best-single speedup、optimized policy 精度、single 方法精度小表和 policy assignment 明细。
- 影响文件：`artifacts/exports/vllm/llama2_7b_018/promising_policy_retest/scripts/eval_promising_policy_quality_vllm.py`，`artifacts/exports/vllm/llama2_7b_018/promising_policy_retest/scripts/eval_promising_policy_quality_vllm_parallel.py`，`artifacts/exports/vllm/llama2_7b_018/promising_policy_retest/scripts/summarize_promising_policy_retest.py`，`artifacts/exports/vllm/llama2_7b_018/promising_policy_retest/quality/`，`artifacts/exports/vllm/llama2_7b_018/promising_policy_retest/summary/`。
- 后续注意：vLLM/lm-eval 子进程在写出完整结果后以 `returncode=-6` 退出，日志显示是进程组清理阶段 `terminate called without an active exception`；每个 job 的 `optimized_policy_quality.csv` 已写出 full `sample_len=1172` 结果并被合并。
