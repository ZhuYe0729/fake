# MIRROR Batch Speed Sweep Plan

## 目标

测试 MIRROR 在不同 batch size 下的端到端 forward 速度，比较：

- 未压缩 `dense_default + AMP`
- 单一方法压缩：`uniform_dense_bf16`、`uniform_dense_nvfp4`、`uniform_sparse_bf16`、`uniform_sparse_nvfp4`
- 我们的混合压缩方法：每个 batch size 从候选混合策略中选择实测速度最快的策略

## 假设

- 未压缩口径沿用当前最终 Pareto 图：`dense_default + AMP`。
- 速度只测试端到端 forward，不重新测试精度。
- 混合方法可针对不同 batch size 选择不同策略；先用已有 speed-aware 策略集合加 `extreme_fastest_microbench` 作为候选，按每个 batch size 实测最快者作为该 batch 的 ours。

## 步骤

1. 生成多 batch size 的 speed sweep selected CSV。
   - verify：CSV 包含 baseline、uniform、mixed candidates。
2. 扩展或新增 batch sweep runner。
   - verify：支持不同 batch size，能输出统一结果表。
3. 使用 GPU 运行 speed validation。
   - verify：每个 batch/method 至少有 forward mean/p50/p90。
4. 汇总每个 batch 的最快 mixed policy，并绘制柱状图。
   - verify：输出 CSV、PNG、PDF，路径位于 `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/`。
