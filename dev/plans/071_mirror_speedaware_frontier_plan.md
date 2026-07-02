# 071 MIRROR Speed-Aware Frontier Plan

## 背景

上一批 analysis-guided 策略主要按 local error 递增，没有充分利用 linear microbench 中不同 module type / shape 的速度差异，导致 Pareto 图不够平滑。新实验改为 speed-aware 策略：优先压缩 MLP 的 gate/up/down，谨慎压缩 attention。

## 步骤

1. 生成 speed-aware 候选策略。
   - 基底为 `dense_bf16`。
   - 主线使用 MLP `sparse_bf16` 数量递增。
   - `dense_nvfp4` 只作为高精度低速补充点。
   - attention 只在 MLP 基本压完后少量加入 `sparse_bf16`。
2. 先跑真实 speed validation。
   - 检查候选点速度是否基本按设计递增。
3. 跑 full quality validation。
   - 覆盖 Chameleon + 8 个 GenImage splits。
4. 汇总新 frontier。
   - 输出 joined/frontier CSV 和 clean PNG/PDF。

## 产物

- `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/`
