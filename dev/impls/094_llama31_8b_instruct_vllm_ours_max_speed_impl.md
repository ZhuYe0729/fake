# 094 Llama-3.1-8B-Instruct vLLM Ours Max-Speed Implementation

## 2026-07-13 - Initial implementation
- 开发目的：为 Llama-3.1-8B-Instruct 建立与 Llama2 max-speed 对齐的策略、导出、测速和 PMPD 评测流程。
- 修改内容：新增基于真实 Llama3.1 GQA fused QKV shape 的 predictor max-speed 策略生成器；新增独立的 phase checkpoint 导出、prefill-only 与 phase-switch 测速、PMPD shard 调度、汇总和使用说明。质量调度默认且校验仅允许 GPU 5、6、7。
- 验证：全部新增 Python 脚本通过 `py_compile`，Shell launcher 通过 `bash -n`；用本地 Llama3.1 配置完成 predictor smoke，确认生成 128-module policy，fused QKV shape 为 `[6144, 4096]`。当前执行环境的 NVIDIA driver 不可通信，故未在此环境启动导出、vLLM smoke、正式测速或 PMPD 作业。
- 影响文件：`artifacts/exports/vllm/ours/llama3.1-8b-instruct/`。
- 后续注意：所有 GPU 任务固定限制在 GPU 5、6、7。
