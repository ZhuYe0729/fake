# 126 Llama3.1 modeling ablation implementation

## 2026-07-22 - Compact quality and latency ablation
- 开发目的：围绕论文的核心建模设计，用一个 Llama3.1 prefill-decode 场景完成紧凑、可复现的消融。
- 修改内容：新增 064 离线实验，复用 057 的固定 54/18 ΔNLL 与 7/5 E2E speed split；实现 local-sum、structure-aware、phase-aware 三档质量代理，以及 Roofline、shape correction、policy calibration 三档速度评估；生成双面板论文图、完整指标和 draft-ready 文本。
- 影响文件：`artifacts/debug/064_llama31_modeling_ablation/`。
- 验证结果：质量 holdout MAE 为 0.0571/0.0468/0.0413；独立 Llama3 local profile MAPE 为 19.99%/8.29%；policy E2E MAPE 为 18.17%/8.30%。完整质量和速度预测均在 1e-8 容差内复现 057，连续两次运行的九类中间数据哈希一致。
- 后续注意：速度消融分别在 local kernel 和 E2E policy 层评价对应组件，不能把 shape correction 的局部作用与 runtime composition overhead 混为同一个 E2E 比较；057/060 主结果保持未修改。

## 2026-07-22 - Detailed paper visualization
- 开发目的：保留紧凑柱状图的同时，增加能够直接观察 prediction 向 measured 对角线收敛的丰富图示。
- 修改内容：新增四面板详细图，分别展示独立 Llama3 module latency、完整 policy E2E latency、holdout ΔNLL scatter，以及四项核心设计的增量误差下降；同步输出独立 caption。
- 影响文件：`artifacts/debug/064_llama31_modeling_ablation/report/modeling_ablation_detailed.{png,pdf}`、`detailed_caption.md`。
- 后续注意：policy-level scatter 明确区分 7 个 train 与 5 个 holdout；quality scatter 只展示 18 个 holdout，避免用训练拟合作为论文证据。

## 2026-07-22 - Roofline and E2E-composition ablation
- 开发目的：补充 Roofline 先验本身的消融，并排除“local shape correction 改善但 E2E 变差”的混合口径干扰。
- 修改内容：在五类 kernel、三个固定种子和 25% 未见 `(N,K)` 组上，以相同 features、ridge 和邻居混合对比 direct log-latency fit、calibrated Roofline、Roofline+shape residual；将论文-facing E2E calibration 从容易产生平台区间常数外推的 PAVA 改为只在 7 个训练 policy 上拟合的正斜率仿射映射，同时保留 PAVA 作为 frozen-result 复现诊断。
- 影响文件：`artifacts/debug/064_llama31_modeling_ablation/scripts/`、`speed/kernel_generalization_*`、`report/ablation_tables.{md,tex}` 及既有报告产物。
- 验证结果：未见 `(N,K)` 的 MAPE 为 14.80%（direct）/10.78%（Roofline）/8.01%（+shape residual）；policy holdout 在同容量正仿射校准下为 8.15%（Roofline）/8.01%（+shape residual），完整模型相对单 BF16 anchor 的 18.17% 明显改善。
- 后续注意：单 BF16 anchor 下 shape-corrected raw sum 的恶化来自缺失的 policy/runtime overhead，不应被解释为 local kernel model 退化；正仿射映射斜率为正，保持原始 policy 速度排序。

## 2026-07-22 - Modeling design and paper-writing document
- 开发目的：将当前速度/精度模型的真实实现、两层速度误差口径、累积消融和论文表述整理为单一可信文档。
- 修改内容：新增中文详细设计说明，给出 phase-aware ReLU 质量公式、method-aware calibrated Roofline、shape residual、正仿射 E2E composition、数据划分、训练超参数、完整消融数值、推荐论文表格和英文 draft-ready 段落。
- 影响文件：`artifacts/debug/064_llama31_modeling_ablation/report/modeling_design_and_ablation.md`。
- 后续注意：明确区分 4657 个 kernel 样本上的 8.00919% MAPE 与 5 个 policy 上的 8.00761% E2E MAPE；057 原有 PAVA 产物未被追溯性修改，064 affine 是修正后的建模消融。

## 2026-07-22 - 调整论文消融表结构
- 开发目的：去除信息价值有限的rank correlation，并显式标明每行属于哪一类建模消融。
- 修改内容：LaTeX表改为 Ablation type / Variant / Error 三列；类型为 Kernel latency、E2E latency 和 Quality prediction；误差单元格显式标明MAPE或MAE。
- 影响文件：064 scripts/build_report.py 和 report/ablation_tables.tex。

## 2026-07-22 - 精简消融分组与Variant命名
- 开发目的：降低表格重复和长名称造成的阅读负担。
- 修改内容：每个ablation type仅在组内首行显示；Variant改为简短的累积设计名；数值列统一为带下降箭头的Error，具体MAPE/MAE口径留给caption或正文解释。
- 影响文件：064 scripts/build_report.py 和 report/ablation_tables.tex。

## 2026-07-22 - 居中消融表并澄清E2E变体
- 开发目的：改善分组对齐，并避免把E2E第三行误解为另一种端到端shape操作。
- 修改内容：使用论文已有的multirow宏包让每个ablation type跨三行居中，tabular三列统一使用居中对齐；E2E第三行改名为 + Refined kernel predictor。
- 语义说明：该行保持相同的正仿射E2E calibration，只把底层纯Roofline cost换成完整Roofline+residual kernel predictor。

## 2026-07-22 - 将E2E消融改为固定最终kernel predictor
- 开发目的：避免把不同评价对象的kernel改进与E2E composition校准串成一条消融链。
- 修改内容：E2E分组缩减为两行，固定最终Roofline+residual kernel predictor，仅比较Raw E2E aggregation与+ E2E calibration；类型名改为Kernel prediction、E2E composition和Quality prediction。
- 验证结果：E2E MAPE从18.17%降到8.01%；该变化只归因于正仿射runtime calibration。

## 2026-07-22 - 将E2E表切换为主实验PAVA校准
- 开发目的：让消融表与057冻结主实验实际使用的monotonic calibration完全一致。
- 修改内容：在speed metrics中加入冻结PAVA预测的显式指标；E2E第二行改为+ Monotonic E2E calibration；同步summary、paper text和设计文档。
- 验证结果：PAVA在5个holdout policy上的MAPE为8.30197%，表中报告8.30%；positive-affine的8.01%仅保留为内部诊断。
