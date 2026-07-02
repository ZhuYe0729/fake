# MIRROR Speed Regression Debug Plan

## 目标

在 `artifacts/debug/032_mirror_speed_regression_debug/` 中单独排查 MIRROR 压缩方法端到端速度与历史结果不一致的问题，明确差异来自：

- 测速脚本/AMP 口径变化
- runtime/kernel 代码变化
- GPU 状态、并发或测量噪声
- 策略文件或输入 batch 变化

## 步骤

1. 汇总历史速度结果和当前复测结果。
   - verify：生成对比 CSV，包含 mean/p50/min/max、脚本、时间、policy、backend_counts。
2. 做代码与策略一致性审计。
   - verify：确认同名 policy 的 module/method 是否完全一致，检查关键 runtime/kernel 文件 git diff。
3. 用同一脚本、同一 GPU、同一 AMP 口径串行复测 batch=16 的关键方法。
   - verify：生成当前统一口径速度表，判断历史差异是否可复现。
4. 输出结论文档。
   - verify：说明哪个口径/实现导致明显变化，以及哪些旧图需要作废或重画。
