## 2026-07-08 - 完成 promising scenario max-speed 策略测试

- 开发目的：为每个 promising scenario 补充不加 P024 精度约束的 `max_speed_hetero` 策略，并实测速度和精度。
- 修改内容：扩展策略求解脚本支持 `--mode max_speed`；复用 hetero 导出流程导出 6 个 unique max-speed checkpoint；扩展 benchmark/quality 脚本支持自定义 checkpoint root、method name 和输出前缀；完成 16 个场景的 vLLM 速度测试和 6 个 unique policy 的 full ARC-Challenge 0-shot 精度测试；将 max-speed 的 speedup、vs best single、acc_norm、policy 加入最终宽表，并补入策略明细。
- 影响文件：`artifacts/exports/vllm/llama2_7b_018/promising_policy_retest/scripts/solve_promising_policies.py`，`benchmark_promising_policy_vllm.py`，`benchmark_promising_policy_vllm_parallel.py`，`eval_promising_policy_quality_vllm_parallel.py`，`summarize_promising_policy_retest.py`，以及 `artifacts/exports/vllm/llama2_7b_018/promising_policy_retest/max_speed/` 和 `summary/`。
- 后续注意：max-speed 是预测 latency unconstrained 最优策略，实际 vLLM 速度并不总是更快；例如 `b8_in16384_out{64,128}` 中 max-speed 实测明显慢于 P024 optimized。精度评测子进程仍在写完结果后出现 vLLM/lm-eval 清理阶段 `returncode=-6`，但所有 job 都写出 `sample_len=1172` 的结果并已合并。

## 2026-07-08 - 补充最终负载选择与 dense bf16 基线

- 开发目的：为最终选定的 1 个 prefill-only 和 2 个 prefill-decoding 场景补充精度、dense bf16 未压缩精度和 dense bf16 实测速度。
- 修改内容：新增最终选择表，记录 dense bf16 median latency、选中策略 median latency、相对 dense/best single 加速、选中策略 ARC-C acc_norm、dense bf16 ARC-C acc_norm 和精度损失。
- 影响文件：`artifacts/exports/vllm/llama2_7b_018/promising_policy_retest/summary/final_workload_selection.md`，`artifacts/exports/vllm/llama2_7b_018/promising_policy_retest/summary/final_workload_selection.csv`。
- 后续注意：dense bf16 精度来自 018 full ARC-C baseline；dense bf16 速度来自 broad-grid vLLM median latency；选中策略精度和速度来自本轮导出 checkpoint 的 vLLM/lm-eval 实测。
