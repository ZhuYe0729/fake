## 2026-07-11 - WikiText phase-NLL debug scaffold
- 开发目的：将精度代理改进实验隔离到 debug 目录，并用 WikiText 通用 NLL 替代下游任务数据进行主建模。
- 修改内容：新增 033 debug 根目录、72 个受控策略（54/18）、300 个 `2048+80` WikiText blocks、逐 policy shard 的 phase NLL evaluator、8 卡按 lane 串行的可恢复 runner、合并器，以及归一化 pooled 与 phase-separated bucket proxy 的对比拟合器。
- 影响文件：`artifacts/debug/033_llama2_7b_chat_wikitext_phase_nll_proxy/`、`dev/plans/086_llama2_wikitext_phase_nll_proxy_debug_plan.md`。
- 验证：静态编译和策略/block shape/split 检查已通过；单 policy full-context WikiText GPU smoke 正在执行，完成后才启动全量采集。
- 后续注意：当前首阶段只验证特征尺度、bucket 化和 phase 拆分；phase-specific local output error 仅在该阶段 holdout 有改善后再叠加，避免同时引入多个变量。

## 2026-07-11 - Full-context budget calibration and stage-1 launch
- 开发目的：用实际 2048-token WikiText 成本校准可接受的全量实验预算。
- 修改内容：300-block dense smoke 成功完成，但约需 20 分钟/策略；主采集改为固定前 100 block（仍为 204,700 prefill 与 8,000 decode 预测 token/policy），并让 shard 合并严格按 `sample_count=100` 过滤，避免混入 smoke 结果。
- 验证：已在 8 张 GPU 上启动 72-policy 第一阶段；每张卡仅一个 7B worker，显存约 15.7GB，未见 OOM。

## 2026-07-11 - Stage-1 normalized proxy result
- 开发目的：逐步验证尺度修复、bucket 化与 phase 拆分各自的影响。
- 修改内容：完成两个场景各 72 个、`sample_count=100` 的 WikiText NLL shard，并自动合并、拟合和输出预测表。
- 验证：normalized pooled prefill-decode holdout Spearman=0.774、MAE=0.738；prefill-only holdout Spearman=0.774、MAE=0.126。phase-separated prefill-decode holdout Spearman=0.263、MAE=9.946，明显劣于 pooled。
- 结论：raw `numel` 放大和层级过参数化是此前失败的主要原因；后续 phase-local-error 实验应保留 pooled `prefill + 80*decode` 目标，而不采用独立 phase 回归的合成预测。

## 2026-07-11 - Phase-specific local-error ablation
- 开发目的：在固定 pooled 目标、策略与 split 下，单独测试 WikiText phase-specific local output error 的增益。
- 修改内容：新增按 phase/method 采集的 bucket×fused-type local output error。发现全模块 hook 会累积 CUDA 临时权重，改为 16-module chunk；又将调度收缩为两 worker/batch，避免 8 份 prepared state 耗尽主机内存。
- 验证：8 个 phase×method 文件均完成。phase-local pooled 的 prefill-only holdout Spearman=0.414（基线 0.774）；prefill-decode=0.684、MAE=0.715（基线 Spearman=0.774、MAE=0.738）。
- 结论：该 16-block、bucket 聚合的 phase-local feature 没有提升排序能力，不纳入最佳代理；保留 normalized pooled generic local-error 作为当前 debug 最优。
