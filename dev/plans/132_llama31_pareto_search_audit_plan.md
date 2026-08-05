# 132 Llama3.1 Pareto search audit plan

## 目标

在不修改现有 057/060/061/068 主实验结果的前提下，为 Llama3.1-8B-Instruct
的 prefill-only 与 prefill-decode 两个论文场景建立小规模经验最优性审计。
候选生成仅采用 FakeVLM 已验证过的 random、neighborhood 和 suspicious
三类方法，真实验证保持当前 Llama3.1 uniform/ours points 的正式口径。

## 实施

1. 在 `artifacts/debug/069_llama31_pareto_search_audit` 下建立独立实验。
2. 每个场景使用固定种子生成 30 个候选：
   - 15 个分层压缩比例 random；
   - 5 个代表性 reference parent × 5%/15% 扰动，共 10 个 neighborhood；
   - 围绕同五个 parent 修改 top-32 高 leverage 模块，共 5 个 suspicious。
3. Prefill-only 复用 068 的 B=8/S=2048 warmed 五次测速及完整
   ARC-Challenge `acc_norm`。
4. Prefill-decode 复用 057 的 B=8/S=2048/O=64 fresh-process 五次测速及
   完整 IWSLT 333 SacreBLEU。
5. 汇总 challenger frontier、ours+challenger union frontier、逐 reference
   domination gap 和论文图。

## 验证

- CPU 检查每场景恰好 15/10/5 个唯一策略、128 个 fused Linear、合法 action
  和稳定 SHA256。
- 每场景各运行一个包含量化/稀疏 action 的 GPU smoke。
- 正式结果要求 60/60 speed 和对应完整任务质量；OOM、GPU 干扰或不完整结果
  不进入汇总。
- 所有最终坐标必须来自真实 E2E speed 与真实下游指标。

## 假设

- 正式运行位于产生 057/061/068 结果的同一 RTX 5090 机器，不涉及 Slurm。
- 复用已有 uniform/ours 质量和速度结果；新增测速前后检查 dense-BF16 anchor，
  漂移超过 3% 时整批重测。
- canonical SparseGPT states 来自 057；禁止 `--prune` 回退和重复 NVFP4 pack。
