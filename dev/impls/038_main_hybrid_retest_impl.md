## 2026-06-06 - Main hybrid retest scaffold
- 开发目的：重新测试 3 个模型、2 个场景下 single/manual/pred 的 linear 聚合延迟和 full-model E2E。
- 修改内容：新增 `scripts/run_main_hybrid_policy_retest.py`，支持生成 single/manual/pred policy、linear summary、full E2E 和 comparison 汇总；新增 plan 文件。
- 影响文件：`scripts/run_main_hybrid_policy_retest.py`、`dev/plans/038_main_hybrid_retest_plan.md`、`artifacts/results/main/001_hybrid_policy_retest/`。
- 后续注意：full-model E2E 覆盖面较大，脚本支持 `--skip-existing-e2e` 用于断点继续。

## 2026-06-06 - Full retest and policy application fix
- 开发目的：完成 3 个模型、2 个场景下 single/manual/pred 的真实 E2E 测试，并保证 manual/pred policy 真正接入 full-model。
- 修改内容：修正 `run_full_e2e` 对 manual/pred 未应用 policy 的问题；single linear summary 改为真实 CUDA linear module 测试；重跑全部 single 结果和修正后的 manual/pred 结果。
- 影响文件：`scripts/run_main_hybrid_policy_retest.py`、`artifacts/results/main/001_hybrid_policy_retest/`。
- 后续注意：Qwen3.5-9B `normal_01` 中 single sparse_bf16 仍略快于 pred，manual shape-level 最优选择在 full-model E2E 上反而较差，需要进一步排查 shape-level linear benchmark 与模型内真实执行上下文的差异。

## 2026-06-06 - Qwen3.5 full-model linear gap tracing
- 开发目的：定位 Qwen3.5-9B `normal_01` 下 standalone linear 测试与 full-model E2E 排名不一致的来源。
- 修改内容：新增 debug 脚本，对 sparse_bf16/manual/pred 的真实模型先做 no-hook E2E，再 hook 每个可压缩 linear 统计模型内 forward 耗时。
- 影响文件：`artifacts/debug/002_qwen35_e2e_linear_gap/scripts/trace_qwen35_policy_gap.py`、`artifacts/debug/002_qwen35_e2e_linear_gap/README.md`。
- 后续注意：hook 会引入同步和整体变慢，`traced_*` 只用于 attribution，最终 E2E 对比仍以 no-hook 结果为准。

## 2026-06-06 - Qwen3.5 gap root-cause analysis
- 开发目的：解释 Qwen3.5-9B `normal_01` 中 manual per-shape oracle 为什么没有稳定赢过 full-model E2E。
- 修改内容：修正 debug hook 目标为顶层替换模块，补充 input shape 记录；验证 full-model sparse_bf16 decode 真实为 `1x1xK` 且约 0.1-0.2ms/call，而当前 standalone manual candidate 对 sparse_bf16 decode 测到 3-4ms/call。
- 影响文件：`artifacts/debug/002_qwen35_e2e_linear_gap/scripts/trace_qwen35_policy_gap.py`、`artifacts/debug/002_qwen35_e2e_linear_gap/ANALYSIS.md`。
- 后续注意：当前 manual benchmark 不是 Qwen full-model oracle；需要改成模型真实 input-rank/call-sequence 的完整 mini-scenario replay。

## 2026-06-06 - Manual module benchmark cold/steady fix
- 开发目的：修正 manual candidate benchmark 将 sparse_bf16 小 m cold-start 重复乘以 output tokens 和 layer count 的问题。
- 修改内容：`benchmark_manual_candidate` 新增 `prefill_first_ms`/`prefill_steady_ms`/`decode_first_ms`/`decode_steady_ms`，并用 group-level 公式估计总延迟；普通 backend 的 cold-start 按 shape group 计一次，`dense_nvfp4_prefill_marlin_decode` 的 lazy materialization 仍按 module 计。
- 影响文件：`scripts/run_main_hybrid_policy_retest.py`、`artifacts/results/main/001_hybrid_policy_retest/manual/normal_01/qwen35-9b_*`、`artifacts/debug/002_qwen35_e2e_linear_gap/ANALYSIS.md`。
- 后续注意：修正后 Qwen3.5-9B `normal_01` manual E2E 为约 `4029.65ms`，仍未超过 pred/sparse_bf16；真正 oracle 需要进一步升级为 full-model 或 model-faithful replay 级别的 policy 评估。

## 2026-06-06 - Qwen3.5 policy ablation scaffold
- 开发目的：用真实 full-model E2E 验证 manual/pred 差异中的 `mlp.down_proj` 和 `self_attn.k/v_proj` 对整体延迟的影响。
- 修改内容：新增 policy ablation debug 脚本，自动生成 manual/pred/sparse 及局部 swap policy，并多次运行 Qwen3.5-9B `normal_01` E2E。
- 影响文件：`artifacts/debug/003_qwen35_policy_ablation/scripts/qwen35_policy_ablation.py`、`artifacts/debug/003_qwen35_policy_ablation/README.md`。
- 后续注意：该实验用于判断 per-linear 额外 cost 是否稳定、可建模。

## 2026-06-06 - Qwen3.5 policy ablation result
- 开发目的：判断 manual/pred 差异是否来自某几个 linear group 的稳定额外 cost。
- 修改内容：完成 9 个 policy 变体各 3 次 full-model E2E；新增 ablation 分析文档。
- 影响文件：`artifacts/debug/003_qwen35_policy_ablation/results/`、`artifacts/debug/003_qwen35_policy_ablation/ANALYSIS.md`。
- 后续注意：hybrid 变体大多集中在 4.02-4.05s，差异接近重复测量方差；`single_sparse_bf16` 呈现 4.68s/3.65s 双峰，说明 global warm state 对结论影响很大，暂不支持简单 per-linear 固定 cost 修正。
## 2026-06-06 - Llama2 normal_02 retest
- 开发目的：新增 `normal_02` 场景，用于测试 `batch_size=1,input_tokens=16384,output_tokens=256` 下 llama2-7b 的 normal 场景结果。
- 修改内容：在主 retest 脚本中新增 `normal_02` 场景，并更新结果目录 README 描述；运行 llama2-7b 的 single/manual/pred 全套真实 E2E 测试。
- 影响文件：`scripts/run_main_hybrid_policy_retest.py`，`artifacts/results/main/001_hybrid_policy_retest/{single,manual,pred}/**/normal_02/`。
- 后续注意：该轮只覆盖 llama2-7b；manual 选择全 `marlin_nvfp4`，pred 在 MLP group 上选择 `dense_nvfp4->marlin_nvfp4`，E2E 上 pred 更快。
