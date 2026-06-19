# artifacts/debug 索引

这个目录保存的是阶段性 debug / ablation / handoff 结果。早期编号里有些结论后来被后续实验修正，下面按编号说明每个目录主要用途和关键内容。

## 001_qwen35_linear_breakdown

Qwen3.5-9B 代表性 Linear 层的单层/多层 kernel 拆解实验，关注 `normal_01` 类长 prefill、短 decode 场景下 `sparse_bf16`、显式 `dense_nvfp4+marlin`、lazy wrapper 的 latency 差异。关键文件是 `README.md`、`FULL_MODEL_TRACE.md` 和 `results/aggregate_breakdown.*`，用于确认若干 Qwen linear 层里 NVFP4/Marlin 路径相对 sparse BF16 的速度优势以及 build/conversion 成本。

## 002_qwen35_e2e_linear_gap

Qwen3.5-9B full-model E2E 与 standalone linear policy 选择不一致的排查包。`scripts/trace_qwen35_policy_gap.py` 会在真实模型里 trace compressible linear，`ANALYSIS.md` 总结 manual/pred policy 的 mismatch；主要价值是定位线性层局部测量和整模型行为之间的 gap。

## 003_qwen35_policy_ablation

围绕 Qwen3.5-9B `normal_01` manual/pred 差异做 full-model policy ablation。它生成 sparse、manual、pred 和局部 swap 变体并跑 E2E timing，`ANALYSIS.md` 是主要结论来源；属于 002 之后的针对性验证。

## 004_llama2_warm_e2e_gap

Llama2-7B `normal_02` warm E2E gap trace。它把真实 manual/pred policy 装进 full model，用 hook 记录 prefill、first decode 和 steady decode，并证明旧 standalone manual benchmark 在 `mlp.gate_proj`、`mlp.down_proj`、`self_attn.o_proj/q_proj` 等组上方向错误。关键输出是 `results/*linear_trace.csv`、`pred_vs_manual_step_group_delta.csv` 和 README 里的 mismatch 表。

## 005_llama2_warm_group_microbench

对 004 发现的问题做更接近真实调用方式的 warmed group microbench：同 shape 的 32 个模块连续执行，去掉 timed loop 里的额外检查和 cache 清理。结果修正了大多数旧 standalone ranking 错误，说明原 manual benchmark 过冷且过于 single-module；`mlp.down_proj` 仍有小 mismatch，需要 full-model ablation。

## 006_llama2_full_model_trace_oracle

尝试用 full-model trace 推导 Llama2 `normal_02` oracle policy，并用 no-hook E2E 验证。初始 trace oracle 仍会误判 q/k/v，后续直接 ablation 发现最佳策略与既有 pred policy 一致：MLP 用 `dense_nvfp4_prefill_marlin_decode`，attention q/k/v/o 用 `marlin_nvfp4`。该目录的结论是 hook trace 能缩小候选，但最终选择仍要靠 no-hook E2E ablation。

## 007_llama2_quality_modeling

Llama2-7B 压缩 policy 的第一版质量建模工作区，收集 module feature、local output error，并跑 ARC-Easy / ARC-Challenge / NLL ablation。关键产物包括 `sensitivity/module_method_errors.csv`、`summary/recommended_proxy_formula.json`、`summary/proxy_correlation.csv`，后续 008-010 的 Pareto 优化主要复用了这里的 `local_rel_mse_log_numel_layer_family` 质量 proxy。

## 008_llama2_pareto_quality_speed

Llama2-7B prefill-only 质量约束速度优化的第一版完整 Pareto 工作区。它用 007 的质量 proxy 和 fresh real-kernel prefill latency 构建候选表，做 constrained optimization，并验证 selected points 的 real E2E prefill latency/NLL/ARC。结论较正面：预测 linear latency 与真实 prefill E2E 排名高度一致，quality cost 与 NLL 排名也较强；这是后续 normal_01/normal_02 扩展的基础。

## 009_llama2_normal01_pareto_handoff

把 008 的 Pareto pipeline 扩展到 Llama2 `normal_01`（prefill 16384、decode 32）的 smoke 包。它证明脚本、policy conversion 和 E2E timing 能跑通，并产生 8 个 frontier points；但 E2E 结果显示短 decode 下 mixed-backend overhead 明显，point 3 甚至比 dense 慢，因此这个目录更像过渡/反例，结论已被 010 的 decode-heavy normal_02 后续工作部分替代。

## 010_llama2_normal02_pareto_handoff

Llama2 `normal_02`（prefill 16384、decode 256）的主要 Pareto 结果包。它修正 009 的统计和 OOM/fragmentation 问题，采用 process-per-repeat E2E 验证，得到 P0-P9 curve，其中 P9 约 `1.22x` measured E2E speedup、NLL delta 约 `0.0368`。关键内容在 `summary/normal02_final_analysis.md`、`summary/llama2_pareto_complete/README.md`、`validation/*`，这是 Llama2 decode-heavy 场景里较可信的一版结果。

## 011_cross_model_normal02_pareto

把 normal_02 Pareto workflow 扩展到跨模型，当前主要完成 Llama3.1-8B 进展。它生成 Llama3.1/Qwen 的 predictor candidates，收集 Llama3.1 sensitivity，构建 frontier，并验证 P0/P7/P9 以及 uniform baselines。结果是一个重要负例：混合 Pareto points 在 Llama3.1 上真实 E2E 反而慢于 dense，但 all-Marlin/all-hybrid uniform 有加速，说明当前 linear-sum latency objective 缺少 model-specific mixed-backend penalty。

## 012_llama2_dialogsum_pareto

用 DialogSum 任务评估 Llama2 `normal02` Pareto 与 uniform baselines 的质量速度关系。它计算 conditional NLL 和 ROUGE-L，并生成 `summary/dialogsum_pareto/*` 图表。结果显示 P8/P9 有可用中高速 tradeoff，但 uniform hybrid 仍是最快端点；同时原 quality proxy 对 DialogSum decode-generation 的 NLL/ROUGE 排名不够可靠，需要重新校准。

## 013_llama2_dialogsum_calibrated_pareto

在 012 基础上用 DialogSum uniform 结果重新校准质量 proxy，并只在 high-speed 区域做 targeted search。C4 是最有价值的新增点，约 `1.176x` speedup 且 NLL/ROUGE 表现较好；但它仍没有支配 uniform hybrid endpoint。关键内容是 `summary/targeted_dialogsum_calibrated/README.md`、`pareto/*`、`quality/targeted_full_card765/*`。

## 014_llama2_prefill_loss_modeling

Llama2 prefill-only loss modeling 的早期版本，目标是用 local linear error 解释 WikiText-2 CE loss delta。但该目录 README 明确标注：它对 NVFP4 的 prefill 质量建模无效，因为用 PyTorch `F.linear` 替换权重，没有经过真实 CUTLASS NVFP4 kernel 的 runtime activation quantization。因此它现在主要作为 offline weight-error 诊断和烟测脚本参考，不应作为最终 NVFP4 Pareto quality model。

## 015_llama2_prefill_kernel_loss_modeling

014 的 kernel-aware 修正版，用真实 `NVFP4Linear` 和 sparse NVFP4 runtime modules 测 dense/sparse NVFP4 的 prefill quality，因此包含 activation quantization。关键输出包括 `sensitivity/module_method_kernel_local_errors.csv`、`ablations/kernel_loss_ablation_*.csv` 和 `summary/kernel_prefill_loss_modeling/*`；这是后续 NVFP4 prefill 质量建模更可信的数据来源。

## 016_llama2_sparse_bf16_precision_proxy

面向 sparse BF16 以及 kernel-aware dense/sparse NVFP4 的 sampled multi-linear precision proxy 拟合实验。它生成 sampled policies、跑 WikiText-2 prefill loss samples，再拟合 proxy 和 holdout trend plots。主要产物在 `policies/`、`loss/`、`model/`、`plots/`、`summary/`，为后续 017 的结构化 proxy 消融提供基础。

## 017_global_coef_structural_ablation

全局系数和结构系数 precision proxy 的消融实验，重点证明仅靠 local error sum 不够，加入 layer-depth 和 linear-type multiplicative coefficients 后能更好预测 downstream loss 差异。当前最有用的结果在 `favorable_multiplicative_pairs/`：11 个 matched sparse NVFP4 pair 上，`final_layer_type` 变体的 MAE/RMSE 最低且方向准确率最高。`stratified_global_ablation/` 也保存了系数、预测和 holdout 图。

## 018_llama2_prefill_global_pareto

基于 017 风格 multiplicative proxy 重建 Llama2 prefill-only Pareto。候选方法默认包含 `dense_bf16`、`dense_nvfp4`、`sparse_bf16`、`sparse_nvfp4`，Marlin 只作为 uniform baseline 汇总而非优化候选。该目录产出了 29 点 frontier、validation、plots 和 compact showcase；推荐展示 P000/P020/P024/P026，核心结果在 `summary/analysis.md`、`showcase/showcase_summary.md` 和 `plots/`。

## 019_dinov3_layerwise_max_speed

DINOv3 layerwise hybrid speed 对比实验，比较现有 hybrid run 与完整重叠 batch size 下的 uniform methods。`hybrid_vs_uniform/README.md` 显示 batch 32 下 hybrid 从 best uniform Sparse BF16 的 `81.607 img/s` 提升到 `86.362 img/s`，约 `1.058x`；收益小于 Llama2 prefill-only，因为 DINOv3 的 ViT projection shape 更统一、uniform Sparse BF16 已接近最优。关键文件是 `hybrid_vs_uniform/*` 和 `code/*`。
