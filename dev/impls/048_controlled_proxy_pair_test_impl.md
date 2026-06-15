## 2026-06-13 - Controlled proxy pair test scaffold
- 开发目的：构造 raw local error 匹配但 final layer/type proxy 差异大的 policy 对，以削弱压缩数量 confound。
- 修改内容：新增 controlled policy 生成和分析脚本；扩展 sparse BF16/NVFP4 loss runner 支持自定义 policy CSV 与 output tag。
- 影响文件：`dev/plans/048_controlled_proxy_pair_test_plan.md`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/scripts/`。
- 后续注意：需要先生成 controlled policies，再用现有 GPU runner 跑 `output-tag=controlled` 的真实 loss。

## 2026-06-13 - Controlled loss run and structural check
- 开发目的：用真实 loss 检验 raw-local-matched pair 是否能证明 layer/type proxy 的必要性。
- 修改内容：完成 sparse BF16、dense NVFP4、sparse NVFP4 的 controlled loss 采样；生成 pairwise delta summary/plot；新增 leave-one-pair-out 结构消融脚本，比较 raw-only、layer、type、layer+type 特征在 controlled pair delta 上的泛化。
- 影响文件：`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/controlled/`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/loss/loss_samples_*_controlled.csv`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/scripts/analyze_controlled_structure_ablation.py`。
- 后续注意：当前 controlled pair 不能证明现有 final coefficients 正确；它主要暴露 raw-only 存在混杂风险，并提示 final coefficients 在 raw-matched counterfactual 上方向不稳。
