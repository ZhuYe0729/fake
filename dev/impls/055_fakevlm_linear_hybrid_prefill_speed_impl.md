## 2026-06-18 - FakeVLM prefill-only linear hybrid speed scaffolding
- 开发目的：新增 FakeVLM prefill-only 场景下 linear layer 粒度混合策略的最大速度测试，覆盖手动 profile 选择和 latency modeling 选择两条路径。
- 修改内容：新增 055 plan 和 `artifacts/debug/021_fakevlm_linear_hybrid_prefill_speed/` 下的 runner、launcher、summary 脚本与 TODO；默认只使用 GPU card 0/1，batch 覆盖 1/2/4/8/16。
- 影响文件：`dev/plans/055_fakevlm_linear_hybrid_prefill_speed_plan.md`、`dev/impls/055_fakevlm_linear_hybrid_prefill_speed_impl.md`、`artifacts/debug/021_fakevlm_linear_hybrid_prefill_speed/`。
- 后续注意：prefill-decode 未实现，保留在 `TODO_prefill_decode.md`；速度最终以真实 FakeVLM prefill forward 计时为准，候选 linear latency 只用于策略选择和解释。

## 2026-06-18 - FakeVLM prefill speed smoke
- 开发目的：验证 021 debug runner 能在当前机器 GPU 上完成最小 FakeVLM prefill-only 速度测试。
- 修改内容：运行 `batch=1,sample_limit=1,warmup=1,iters=1,manual_iters=1` smoke，覆盖 `manual_profile`、`latency_model`、`uniform_dense_bf16`；生成候选表、policy、速度 CSV 和 summary。
- 影响文件：`artifacts/debug/021_fakevlm_linear_hybrid_prefill_speed/candidates/`、`policies/`、`speed/prefill_speed.csv`、`summary/prefill_speed_summary.*`、`logs/batch_1.log`、`status/batch_1.json`。
- 后续注意：smoke 结果仅用于链路验证；完整结论需要默认 batch `1 2 4 8 16` 和完整 family 集合重跑。

## 2026-06-18 - 完整 batch sweep 结果检查
- 开发目的：检查用户完成的完整 FakeVLM prefill-only batch sweep，并修正 batch 1 混入 smoke 旧行的问题。
- 修改内容：发现 batch 1 的部分 family 因未设置 `OVERWRITE=1` 仍沿用 smoke `warmup=1,iters=1`；使用 `BATCH_SIZES=1 OVERWRITE=1` 重跑 batch 1 后重新生成 summary，所有 batch/family 统一为 `warmup=3,iters=10`。
- 影响文件：`artifacts/debug/021_fakevlm_linear_hybrid_prefill_speed/speed/prefill_speed.csv`、`summary/prefill_speed_summary.csv`、`summary/prefill_speed_summary.md`、batch 1 的 candidates/policies/log/status。
- 后续注意：最终 summary 中 batch 1/2/4/8/16 均可作为同口径结果；`speed/prefill_speed.csv` 保留了历史追加行，读取时应使用 summary 或按 timestamp 取最新。

## 2026-06-18 - 结果文档和分析
- 开发目的：解释 FakeVLM hybrid 相比 best uniform 加速较小的原因，并沉淀完整结果、策略和分析。
- 修改内容：新增 `ANALYSIS.md`，整理 E2E 速度、uniform baseline、manual/model 策略组成、linear-only aggregate 与 E2E 速度差异，并说明与 LLaMA prefill-only 的差别。
- 影响文件：`artifacts/debug/021_fakevlm_linear_hybrid_prefill_speed/ANALYSIS.md`、`artifacts/debug/021_fakevlm_linear_hybrid_prefill_speed/README.md`、`dev/impls/055_fakevlm_linear_hybrid_prefill_speed_impl.md`。
- 后续注意：正式引用时优先使用 `ANALYSIS.md` 和 `summary/prefill_speed_summary.csv`；`speed/prefill_speed.csv` 中包含历史追加行。

## 2026-06-18 - 补充 LLaMA/Qwen 对比和 linear 占比
- 开发目的：回应 FakeVLM 相比 LLaMA/Qwen 看起来收益偏小的问题，统一 baseline 口径并分析 linear/non-linear 延迟占比。
- 修改内容：在 `ANALYSIS.md` 中加入 `001_hybrid_policy_retest` 的 LLaMA-2/Qwen3.5 prefill-only 结果、normal prefill-decode 示例、策略组成，以及 LLaMA/Qwen/FakeVLM 的 selected-linear 与 non-linear/unmodeled 延迟占比表。
- 影响文件：`artifacts/debug/021_fakevlm_linear_hybrid_prefill_speed/ANALYSIS.md`、`dev/impls/055_fakevlm_linear_hybrid_prefill_speed_impl.md`。
- 后续注意：同口径下 FakeVLM prefill-only 相比 best uniform 的收益不低于 LLaMA/Qwen；之前“很明显”的观感主要来自 vs dense BF16 或带 decode 的 normal 场景。

## 2026-06-18 - 修正 LLaMA/Qwen pure prefill 对比口径
- 开发目的：按用户指出的 `artifacts/results/benchmarks/hybrid/manual/prefill_only/` 主结果修正 LLaMA/Qwen pure prefill 对比，避免误用 `001_hybrid_policy_retest` 的旧 retest 结果。
- 修改内容：`ANALYSIS.md` 改为引用 manual pure-prefill linear aggregate：LLaMA-2 hybrid `2.1945x`、best uniform `1.9406x`、hybrid/best `1.1309x`；同时明确该目录是 module-level linear 汇总，而 FakeVLM 021 是 full-model prefill forward。
- 影响文件：`artifacts/debug/021_fakevlm_linear_hybrid_prefill_speed/ANALYSIS.md`、`dev/impls/055_fakevlm_linear_hybrid_prefill_speed_impl.md`。
- 后续注意：若要完全同口径比较，需要给 FakeVLM 也补一个 linear-only aggregate 表，或给 LLaMA/Qwen 跑真实 full-model prefill forward；目前文档明确标注了 scope 差异。
