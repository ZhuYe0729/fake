# DialogSum Llama2 Eval Plan

## Summary
实现一个临时 DialogSum 生成式评测脚本，先仅支持 `llama2-7b` 端到端跑通：加载 DialogSum validation/test，使用固定 instruction prompt，调用真实 `model.generate()` 生成 summary，并计算 ROUGE-1/2/L。

## Assumptions
- `llama2-7b` 模型权重优先使用仓库现有路径 `/home/agent/wja/data/models/LLM-Research/llama-2-7b`，也允许通过参数覆盖。
- 计算节点无网络，因此脚本支持 `local_files_only`，依赖 Hugging Face dataset/model 已提前缓存。
- 临时脚本以小样本 smoke test 为默认用途，不在登录节点实际运行 7B GPU 推理。

## Key Changes
- 新增 `scripts/temp_eval_dialogsum_llama2.py`。
- 默认使用 prompt：`Summarize the following dialogue.\n\n{dialogue}\n\nSummary:`。
- 使用 greedy decoding，`max_new_tokens=128`。
- 保存逐样本 `results.jsonl` 和聚合 `summary.json`。

## Test Plan
- `python -m py_compile scripts/temp_eval_dialogsum_llama2.py` 检查语法。
- 用 `--help` 检查 CLI 参数可用。
- 端到端 GPU 运行由集群计算节点执行。
