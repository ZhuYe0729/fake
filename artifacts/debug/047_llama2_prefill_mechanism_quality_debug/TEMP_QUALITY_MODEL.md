# 临时说明：当前 prefill-only 精度 / 质量建模

> 状态：`047` 是对 `046` 主实验的隔离调试，下面的 v2 模型**尚未**用于帕累托重求解或主结果。本文的“精度”在建模时严格指 real-vLLM NLL；下游任务分数只作为最终外部验证，不参与拟合。

## 1. 预测目标与口径

给定一个 prefill-only 层异构策略 \(p\)，预测

\[
  \Delta \operatorname{NLL}(p) = \operatorname{NLL}_{\rm vLLM}(p)-\operatorname{NLL}_{\rm vLLM}(\text{dense BF16}).
\]

- 标签是 vLLM 直接返回的 prompt log-prob，使用固定的 100 段 WikiText、每段 2048 个评分 token，总计 204,800 token。
- 每个策略均通过 phase-heterogeneous 导出后，由真实 vLLM `phase_hetero_mytest` 推理；NVFP4 因而包含实际的激活量化路径。
- 本轮策略只改变 prefill。所有 decode method 固定为 dense BF16，因此该模型不包含 decoding 误差。
- dense BF16 策略的预测值被约束为 0；没有可学习的截距。这使预测值的含义与质量约束一致。

这里使用 NLL 而非下游任务分数，是因为它可在固定、低成本的无标签校准样本上稳定采样；NLL 到下游分数的关系需要在帕累托点完成后单独验证，而不是被拟合进代理模型。

## 2. 输入：模块局部扰动表

输入来自 `artifacts/debug/007_llama2_quality_modeling/sensitivity/module_method_errors.csv`。对每一层 \(l\) 和 fused-linear 类型 \(t\)，先将原始子模块局部相对 MSE 平均为：

- `qkv_proj`：`q_proj`、`k_proj`、`v_proj` 的均值；
- `o_proj`：`o_proj`；
- `gate_up_proj`：`gate_proj`、`up_proj` 的均值；
- `down_proj`：`down_proj`。

模型共有 32 层，按每 8 层分为四个 layer bucket \(g\in\{0,1,2,3\}\)，模块类型 \(t\in\{\mathrm{qkv,o,gateup,down}\}\)。与早期模型不同，v2 **不在 bucket 内去重**：每一个被改变的 fused-linear 都贡献其自身的局部误差。这一点对表达层异构选择是必要的。

## 3. 将压缩方法拆成两种机制

对于策略中每个 fused-linear，v2 不把 `sparse_nvfp4` 当成独立、不可解释的第四种误差来源，而拆为两个自然通道：

| prefill 方法 | 量化通道 \(Q\) | 稀疏通道 \(S\) |
|---|---:|---:|
| `dense_bf16` | 0 | 0 |
| `dense_nvfp4` | dense-NVFP4 局部误差 | 0 |
| `sparse_bf16` | 0 | sparse-BF16 局部误差 |
| `sparse_nvfp4` | dense-NVFP4 局部误差 | sparse-BF16 局部误差 |

因此，对 bucket/type 聚合后得到 \(Q_{g,t}(p)\) 和 \(S_{g,t}(p)\)。训练前将二者同时除以一个固定尺度 \(z\)：训练集所有策略的 \(\sum_{g,t} Q_{g,t}+S_{g,t}\) 的平均值。这个缩放只改善数值优化；不会改变策略排序所代表的机制。

## 4. 当前 v2 公式

令 \(Q_g=\sum_t Q_{g,t}\)，\(S_g=\sum_t S_{g,t}\)。模型为：

\[
\widehat{\Delta\mathrm{NLL}}(p)=
  \sum_{g,t} w^Q_{g,t}Q_{g,t}
  +\sum_{g,t} w^S_{g,t}S_{g,t}
  +\sum_g a_g S_g^2
  +\sum_g c_gS_gQ_g.
\]

所有系数 \(w^Q,w^S,a,c\geq0\)，共 40 个：

- `wQ`：量化在不同层区间和模块类型的局部敏感度；
- `wS`：稀疏的对应局部敏感度；
- `a`：同一 layer bucket 中稀疏误差随使用量积累的非线性惩罚；
- `c`：稀疏和量化同时出现时的额外交互惩罚。

`sparse_nvfp4` 同时进入 Q 和 S，因此会自然受到局部项、稀疏累积项和 Q×S 交互项，而无需人为给它一个独立的方法标签。

### 非负参数化的已修正点

初版 v2 使用 `softplus(raw)`。这是不对的：`softplus(0)=0.693`，故“raw=0”并不等于零系数；L2 又会把系数推回这个正基线，导致低敏感策略被凭空惩罚。

当前实现使用 `ReLU(raw)`，以 `0.01` 初始化。这样系数可学习为严格 0，满足 dense-BF16 零锚点，也仍保证预测不会因压缩机制而出现人为的负损失。

训练使用 Adam（5000 步、学习率 0.025），目标为训练样本的均方误差加 `1e-4` L2 正则。

实现：[fit_mechanism_proxy.py](scripts/fit_mechanism_proxy.py)。冻结后的系数在 [report/model.json](report/model.json)。

## 5. 校准数据与严格 holdout

训练/验证都使用同一 NLL 协议，但策略彼此独立：

| 来源 | 训练 | holdout | 用途 |
|---|---:|---:|---|
| `046` 已有策略 | 54 | 18 | 覆盖既有统一/混合策略分布 |
| `047` 新机制策略 | 12 | 6 | 检验量化、稀疏及其组合是否被正确区分 |

新 18 个策略有四个 family：

- quant-only：24/48/72/96（训练），112/120（holdout）个经过局部敏感度排序的 dense-NVFP4 fused-linear；
- sparse-only：2/4/8/16（训练），32/64（holdout）个 sparse-BF16 fused-linear；
- co-located：2/8（训练），24（holdout）个 sparse-NVFP4 fused-linear；
- separated：固定 80 个 dense-NVFP4，再加入 2/8（训练）或 24（holdout）个 sparse-BF16。

全部策略、数量、哈希和 split 位于 [manifest.json](manifest.json)，标签位于 [nll.csv](nll.csv)。脚本会同时验证策略 SHA-256、样本 SHA-256、block 数和 token 数，避免把错误导出或不同样本混进训练。

## 6. 与原 v1 的区别

| 方面 | `046` v1 | `047` 当前 v2 |
|---|---|---|
| 局部特征 | 方法 × bucket × type 的单通道聚合；Llama2 bucket 内去重 | 每个改变的 fused-linear 累加；Q 与 S 分通道 |
| 方法关系 | NVFP4、稀疏作为离散方法标签 | `sparse_nvfp4 = Q + S`，再显式建模积累/交互 |
| 全局项 | 正的 global/method/bucket/type 因子及可学习 bias | 非负 Q/S 细粒度系数、\(S^2\)、\(SQ\)，无 bias |
| 标签 | real-vLLM NLL | 同一 real-vLLM NLL 口径 |
| 当前定位 | 原主实验模型 | 隔离调试模型，尚不可替代 v1 |

## 7. 当前验证结论与限制

ReLU 零锚点修正后，原 `046` holdout 有改善：

| 指标 | v1 | 当前 v2 |
|---|---:|---:|
| MAE | 0.1214 | 0.0849 |
| RMSE | 0.1493 | 0.1113 |
| Spearman | 0.8204 | 0.8741 |

但新增机制 holdout 的 MAE 仍为 0.2145，且总体仍有高估。根因不是 NLL 标签不真实：例如按局部敏感度选择后，120/128 模块量化的实测 ΔNLL 约为 `3e-4`。根因是当前特征只使用“已选模块误差的总和”，尚未显式表示**剩余未压缩模块所保护的关键敏感度**或选择分布。

因此当前 v2 不满足“用于约束求解”的标准。下一步应保持 Q/S 机制拆分不变，并以自然方式补充策略选择分布的信息（例如未压缩模块的敏感度质量、或等价的选中误差分位/分布特征），然后以现有的 18 项机制 holdout 决定是否接受。不能仅以训练误差或旧 holdout 改善为理由重求解。

## 8. 与速度模型的关系

本文件只描述精度/质量模型。速度仍使用既有 `KernelLatencyPredictor` 的 roofline/kernel 基模型，并通过已完成的 E2E 校准将 raw linear latency 映射到实际 prefill 时间。精度模型的输出是质量约束，速度模型的输出是优化目标；两者在策略求解时组合，但此处没有把速度数据拟合进 NLL。
