## 2026-06-05 - Llama predictor policy full E2E
- 开发目的：为 Llama-2-7B 与 Llama-3.1-8B 接入 predictor hybrid policy，并运行完整模型端到端测试。
- 修改内容：
  - 新增 `fake/models/llama_kernels.py`，支持从通用 offline policy JSON 替换 Llama compressible Linear。
  - 新增 `scripts/bench_llama_predictor_hybrid_e2e.py`，加载完整 Llama 模型、应用 predictor policy、运行 prefill-only/normal 场景。
  - 生成 `artifacts/results/benchmarks/hybrid/pred/llama_predictor_hybrid_full_e2e.csv`。
  - 更新 `artifacts/results/benchmarks/hybrid/pred/README.md`，加入 Llama full model E2E 表。
- 影响文件：
  - `fake/models/llama_kernels.py`
  - `scripts/bench_llama_predictor_hybrid_e2e.py`
  - `artifacts/results/benchmarks/hybrid/pred/llama_predictor_hybrid_full_e2e.csv`
  - `artifacts/results/benchmarks/hybrid/pred/README.md`
- 测试结果：
  - Llama-2-7B normal_01：224/224 replaced，E2E 2254.3754 ms。
  - Llama-3.1-8B normal_01：224/224 replaced，E2E 2160.8251 ms。
  - Llama-2-7B prefill_only：224/224 replaced，prefill 1172.6062 ms。
  - Llama-3.1-8B prefill_only：224/224 replaced，prefill 1215.0254 ms。
- 后续注意：prefill_only 的旧 manual artifact 是 linear/module-level 口径，本次是 full model forward，不能直接按毫秒值解读为同一口径。
## 2026-06-05 - 整理 predictor hybrid 结果与对比分析
- 开发目的：按 manual 目录的场景结构重新组织 predictor hybrid 输出，并补充策略差异和端到端可比性分析。
- 修改内容：在 `pred/prefill_only/` 和 `pred/normal_01/` 下拆分策略、summary、module timing、Llama full E2E CSV；新增两个场景 summary 和总分析文档；更新 pred README 的目录说明。
- 影响文件：`artifacts/results/benchmarks/hybrid/pred/` 下的场景子目录、`Predictor_Hybrid_Analysis.md`、`README.md`，以及本实现记录。
- 后续注意：normal 场景下 Llama full E2E 与 manual 仍存在 benchmark 路径差异；Qwen3.5 normal predictor policy 还需要 full-model E2E 验证。

## 2026-06-05 - 补充 Qwen3.5 normal predictor full E2E
- 开发目的：补齐 Qwen3.5-9B normal 场景 predictor policy 的真实 full-model E2E 测试。
- 修改内容：为 Qwen E2E 脚本增加 `--methods`、`--output-csv`、`--scenario-name`；修复 Qwen predictor policy 在 full model 中的 suffix 名称匹配；补跑 Qwen3.5-9B normal predictor_hybrid，替换 248/248 个 linear。
- 影响文件：`scripts/bench_qwen3_5_swh_e2e.py`、`fake/models/qwen3_5_kernels.py`、`artifacts/results/benchmarks/hybrid/pred/normal_01/qwen3_5_predictor_hybrid_full_e2e.csv`、`predictor_hybrid_full_e2e.csv`、相关 summary/analysis 文档。
- 后续注意：Qwen3.5 normal predictor full E2E 为 4204.76ms，慢于 manual hybrid 3308.00ms；当前 predictor 对 W4A4/W4A16 路径的模型级 decode/materialization 成本估计不足。
