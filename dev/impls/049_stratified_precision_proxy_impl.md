## 2026-06-13 - Stratified proxy sampling and evaluation
- 开发目的：用固定 count、raw local error 分位 bin、layer/type composition 多样性的采样，降低 raw-only 与压缩数量/误差总量的混杂。
- 修改内容：新增 stratified policy generator；扩展 `fit_proxy_ablation.py` 支持自定义 method policy 模板、loss tag、输出子目录和样本数；完成 sparse BF16、dense NVFP4、sparse NVFP4 各 80 条 stratified loss；新增 count/raw residualized proxy 分析。
- 影响文件：`dev/plans/049_stratified_precision_proxy_plan.md`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/scripts/generate_stratified_proxy_policies.py`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/scripts/fit_proxy_ablation.py`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/scripts/analyze_stratified_residual_proxy.py`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/stratified/`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/stratified_ablation/`。
- 后续注意：stratified holdout 显示 sparse NVFP4 的 final layer/type 最稳定；dense NVFP4 仍需要非线性校准；sparse BF16 在这组数据中主要由 count/raw 解释，结构项收益不明显。
