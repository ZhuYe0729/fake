## 2026-06-11 - DialogSum llama2 temporary eval
- 开发目的：实现 `llama2-7b` 在 DialogSum 上的生成式端到端 smoke/eval 脚本。
- 修改内容：新增临时评测脚本，加载 DialogSum split，固定 instruction prompt，调用 `model.generate()` 贪心生成，计算 ROUGE-1/2/L，并输出 `results.jsonl` 与 `summary.json`。
- 影响文件：`scripts/temp_eval_dialogsum_llama2.py`、`dev/plans/044_dialogsum_llama2_eval_plan.md`。
- 后续注意：真实 7B 推理需要在 GPU 计算节点运行；离线环境需提前缓存模型和数据集。

## 2026-06-11 - Run test split on cuda:5
- 开发目的：按要求在 `cospaq` 环境使用 `cuda:5` 直接运行 DialogSum test split smoke 评测。
- 修改内容：将脚本默认 split 改为 `test`；运行 `llama2-7b` 生成式评测默认 16 条样本。
- 影响文件：`scripts/temp_eval_dialogsum_llama2.py`、`artifacts/results/dialogsum_llama2_7b/temp_eval/`。
- 后续注意：本次为默认 quick smoke 的 16 条样本；全量 test 可用 `--limit -1` 运行。
