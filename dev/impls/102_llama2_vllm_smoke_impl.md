## 2026-07-15 - Llama2 vLLM 单请求 smoke 验证
- 开发目的：在不依赖评测数据集的条件下验证原始模型推理链路。
- 修改内容：新增文件式 `smoke_vllm_generate.py`，避免 vLLM multiprocessing spawn 无法从 stdin 启动的问题。
- 影响文件：`artifacts/exports/vllm/baselines/llama2-7b-chat/scripts/smoke_vllm_generate.py`。
- 验证：本地 Llama-2-7B-Chat 成功加载并生成正常文本；尚未发现可运行的 Llama2 uniform 压缩 checkpoint。
