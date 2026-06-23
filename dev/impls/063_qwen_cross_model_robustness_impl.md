## 2026-06-23 - Initial cross-model scaffold
- 开发目的：建立 Qwen3.5 多尺寸模型下的 uniform 与 linear hybrid cross-model 鲁棒性速度测试流程。
- 修改内容：新增计划文件和 027 debug artifact 目录；实现单任务测速 runner、任务级多 GPU launcher、cross-model summary 生成入口。
- 影响文件：`dev/plans/063_qwen_cross_model_robustness_plan.md`，`dev/impls/063_qwen_cross_model_robustness_impl.md`，`artifacts/debug/027_qwen_cross_model_robustness/`。
- 后续注意：默认不跑 Qwen3.5-27B；完整速度数据需要在 GPU 上运行 launcher。

## 2026-06-23 - Full cross-model run
- 开发目的：执行 Qwen3.5 0.8B/2B/4B/9B 在 3 个 workload、7 个方法下的完整速度测试。
- 修改内容：使用 GPU 5/6/7 完成 84/84 个组合测试，并生成 cross-model summary 表。
- 影响文件：`artifacts/debug/027_qwen_cross_model_robustness/speed/qwen_cross_model_raw.csv`，`artifacts/debug/027_qwen_cross_model_robustness/summary/`，`dev/impls/063_qwen_cross_model_robustness_impl.md`。
- 后续注意：当前结果显示 `our_linear_hybrid` overall geomean 为 `1.028x`，优于 best transferred uniform 的 `0.985x`。

## 2026-06-23 - Llama additions
- 开发目的：在同一 027 cross-model 实验中补充 Llama-2-7B 与 Llama-3.1-8B。
- 修改内容：扩展 runner 支持 Llama 模型加载、uniform kernel 替换和 Llama predictor hybrid；补跑 42/42 个 Llama 组合并重新生成 summary。
- 影响文件：`artifacts/debug/027_qwen_cross_model_robustness/scripts/`，`artifacts/debug/027_qwen_cross_model_robustness/speed/qwen_cross_model_raw.csv`，`artifacts/debug/027_qwen_cross_model_robustness/summary/`，`dev/impls/063_qwen_cross_model_robustness_impl.md`。
- 后续注意：加入 Llama 后 `our_linear_hybrid` overall geomean 为 `1.136x`，优于 best transferred uniform 的 `1.064x`。

## 2026-06-23 - Dense policy wrapper fix
- 开发目的：修正 predictor hybrid 在 `dense_bf16/dense_bf16` 策略下仍替换成 hybrid wrapper，导致小模型 normal 场景慢于 dense baseline 的问题。
- 修改内容：Qwen/Llama predictor hybrid 遇到 `dense_bf16/dense_bf16` 时保留原始 `nn.Linear`，仅记录 backend count；重跑 18 个 `our_linear_hybrid` 组合并刷新 summary。
- 影响文件：`fake/models/qwen3_5_kernels.py`，`fake/models/llama_kernels.py`，`artifacts/debug/027_qwen_cross_model_robustness/speed/qwen_cross_model_raw.csv`，`artifacts/debug/027_qwen_cross_model_robustness/summary/`，`dev/impls/063_qwen_cross_model_robustness_impl.md`。
- 后续注意：修正后 `our_linear_hybrid` overall geomean 为 `1.195x`；少数单 workload 仍低于 dense，原因是 predictor 选择了部分非 dense backend，属于预测口径非 E2E oracle 问题。
