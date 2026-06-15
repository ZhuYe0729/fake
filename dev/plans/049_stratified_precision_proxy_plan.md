# 049 Stratified Precision Proxy Plan

## 目标

用更可靠的分层采样替代 adversarial controlled pair：固定压缩数量，在 raw local error 分位 bin 内采样 layer/type composition 多样的 policies，用真实 loss 重新拟合并评估 sparse BF16、dense NVFP4、sparse NVFP4 的精度代理。

## 假设

- sparse BF16 继续使用 014 的 sparse local error，dense/sparse NVFP4 使用 015 的 kernel-aware local error。
- 每个 method 单独生成 stratified policy，因为 local error 分布不同。
- loss runner 已支持 `--policies-csv` 和 `--output-tag`，因此无需改真实推理逻辑。

## 实施步骤

1. 新增 stratified policy generator  
   验证：为三个 method 生成 policy CSV 和 metadata，确保每个 count/raw-bin 有指定数量样本。

2. 扩展 proxy ablation fitting 支持自定义 policy/loss/tag/output  
   验证：能在不影响原 `ablation/` 的情况下读取 `loss_samples_{method}_stratified.csv` 并输出到 `stratified_ablation/`。

3. 生成 stratified policies 并跑真实 loss  
   验证：每个 method 的 stratified loss CSV 行数与 policy 行数一致。

4. 拟合与画图  
   验证：输出 holdout prediction plots 和 summary，用于判断 raw-only 与 layer/type/final 的差异。
