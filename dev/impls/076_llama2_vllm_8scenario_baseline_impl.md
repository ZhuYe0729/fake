## 2026-07-07 - 初始化 selected-8 vLLM baseline

- 开发目的：实现 Llama2-7B selected 8 scenarios 的 vLLM 速度与精度 baseline。
- 修改内容：添加计划文件，准备补充导出、benchmark、精度评测和汇总脚本。
- 影响文件：`dev/plans/076_llama2_vllm_8scenario_baseline_plan.md`，`dev/impls/076_llama2_vllm_8scenario_baseline_impl.md`。
- 后续注意：正式结果必须来自实际 vLLM checkpoint 推理；hetero 精度按 3 个唯一策略复用。

## 2026-07-07 - 添加 selected-8 vLLM 脚本

- 开发目的：补齐 handoff 文档 8 个负载场景下的导出、速度测试、精度测试和汇总入口。
- 修改内容：新增 selected-8 checkpoint 导出脚本、vLLM benchmark 脚本、lm-eval/vLLM ARC-Challenge 精度脚本、汇总脚本和一键运行脚本。
- 影响文件：`artifacts/exports/vllm/llama2_7b_018/scripts/` 下的 selected-8 脚本，`artifacts/exports/vllm/llama2_7b_018/summary/selected_8_scenarios_speed_quality.*`。
- 后续注意：当前机器 GPU 为 `Exclusive_Process` 且 PyTorch 创建 CUDA context 返回 `cudaErrorDevicesUnavailable`；实际导出/测试需先解决 GPU compute mode 或设备访问限制。

## 2026-07-07 - 增加多卡并行 launcher

- 开发目的：支持每张 GPU 跑一个独立 vLLM 测试任务，避免同卡多任务干扰速度结果。
- 修改内容：新增速度并行 launcher，按 method 分配 GPU 并合并 job CSV；新增 hetero 精度并行 launcher，按 strategy 分配 GPU 并合并质量 CSV；一键脚本默认调用并行 launcher。
- 影响文件：`benchmark_selected8_vllm_parallel.py`，`eval_selected8_quality_vllm_parallel.py`，`run_selected8_vllm_baseline.sh`。
- 后续注意：launcher 只通过 `CUDA_VISIBLE_DEVICES` 隔离进程，不做模型多卡切分；GPU compute mode 仍需外部权限修复后才能实际运行。

## 2026-07-07 - 完成导出与 selected-8 基线测试

- 开发目的：产出 selected 8 scenarios 的 vLLM 速度和 full ARC-Challenge 精度 baseline。
- 修改内容：导出 `uniform_marlin_nvfp4` 和 `hetero_strategy_{a,b,c}`；并行完成 6 个方法在 8 个场景下的 vLLM 速度测试；并行完成 3 个 hetero strategy 的 full ARC-Challenge 精度测试；生成最终汇总表。
- 影响文件：`artifacts/exports/vllm/llama2_7b_018/{uniform_marlin_nvfp4,hetero_strategy_a,hetero_strategy_b,hetero_strategy_c}`，`benchmarks/selected_8_scenarios_vllm/`，`quality/selected_8_scenarios/`，`summary/selected_8_scenarios_speed_quality.*`。
- 后续注意：当前 shell 需要显式清理 `_CUDA_COMPAT_STATUS` 才能稳定访问 GPU；vLLM/lm-eval worker 退出阶段有 abort，但三个 hetero quality job 均已写出完整 ARC-Challenge 结果。

## 2026-07-07 - 补充场景配置说明

- 开发目的：让最终 summary 文档自包含 8 个负载场景的具体配置。
- 修改内容：在汇总脚本和生成的 Markdown 中增加 `Scenario configs` 表，包含 batch、input_len、output_tokens、prefill_M 和 hetero strategy。
- 影响文件：`summarize_selected8_vllm.py`，`summary/selected_8_scenarios_speed_quality.md`。
- 后续注意：后续重新运行汇总脚本会保留该场景配置表。
