# 速度与精度建模设计及消融实验说明

> 本文档记录当前实现中可用于论文表述的建模方法与消融结果。实验对象为
> Llama3.1-8B-Instruct，场景为 prefill-decode，batch size 为 8，输入长度为
> 2048，输出长度为 64。所有消融复用 057 中冻结的 RTX 5090 测量，不修改
> 057/060 的主实验结果。

## 1. 问题定义

对每个可压缩 Linear module $i$，策略分别选择 prefill 和 decode 阶段的执行方法：

$$
a_i = \left(a_i^{\mathrm{p}}, a_i^{\mathrm{d}}\right).
$$

完整策略记为 $\pi=\{a_i\}_{i=1}^{M}$。候选执行方法包括 dense BF16、dense
NVFP4、SparseGPT-calibrated sparse BF16、SparseGPT-calibrated sparse NVFP4
和 Marlin W4A16。目标是在不对组合空间中的所有策略进行真实 vLLM 测量的情况下，
预测策略的质量损失和执行时间，并求解：

$$
\begin{aligned}
\min_{\pi}\quad & \widehat L_{\mathrm{E2E}}(\pi) \\
\mathrm{s.t.}\quad & \widehat{\Delta\mathrm{NLL}}(\pi) \leq \epsilon, \\
& \pi \in \mathcal A_{\mathrm{runtime}}.
\end{aligned}
$$

$\mathcal A_{\mathrm{runtime}}$ 表示 backend 支持的合法 kernel、shape 和 phase
组合。扫描不同的质量预算 $\epsilon$ 得到预测候选，再对少量候选进行真实 vLLM
速度、NLL 和下游任务闭环测试。论文最终 Pareto 图使用实测值，而不是直接使用预测值。

## 2. 精度建模

### 2.1 监督目标：真实 phase-vLLM WikiText NLL

精度代理的监督信号是固定 WikiText blocks 上的 teacher-forced NLL。dense BF16
和每个异构策略使用相同 token、相同 phase-vLLM backend 和相同执行语义。策略的监督
标签为：

$$
y(\pi)
= \Delta\mathrm{NLL}(\pi)
= \mathrm{NLL}(\pi)-\mathrm{NLL}(\pi_{\mathrm{BF16}}).
$$

当前 Llama3.1 prefill-decode 实验使用100个固定样本，每个样本在2048-token
context 后评价64个 teacher-forced token，共6400个目标 token。它会真实经历一次
prefill 和后续 decode transition，因此监督标签已经包含两阶段共同造成的误差。

需要强调：质量标签不是人为构造的
$\Delta\mathrm{NLL}_{\mathrm{prefill}}+63\Delta\mathrm{NLL}_{\mathrm{decode}}$。
阶段影响由真实 runtime 标签以及下面的 phase-aware 特征共同学习。

### 2.2 Canonical 压缩状态

局部误差与策略 NLL 必须来自和真实推理一致的压缩语义：

- sparse BF16 使用 SparseGPT 校准后的 2:4 权重；
- sparse NVFP4 使用 SparseGPT pairwise 4:8 状态，最终只在 phase exporter 中做一次
  NVFP4 packing；
- dense/sparse NVFP4 均经过真实 activation quantization runtime；
- 禁止对已经量化的权重重复 packing，也禁止使用简单 `--prune` 代替 SparseGPT；
- uniform 和 heterogeneous policy 使用相同的 phase-heterogeneous vLLM runtime。

这些约束并不是精度模型的额外设计，而是保证训练标签和最终推理语义一致的必要条件。

### 2.3 局部误差特征

对于 phase $r$、layer bucket $b$、模块类型 $t$ 和压缩方法 $m$，记录相对于
dense BF16 输出的局部相对均方误差：

$$
e_{r,b,t,m}
=
\frac{\lVert Y_{r,b,t,m}-Y_{r,b,t,\mathrm{BF16}}\rVert_2^2}
{\lVert Y_{r,b,t,\mathrm{BF16}}\rVert_2^2+\varepsilon}.
$$

当前特征使用：

- 两个 phase：prefill、decode；
- 五种执行方法；
- 四个 layer bucket，每个 bucket 包含连续8层；
- 四种 fused module type：`qkv_proj`、`o_proj`、`gate_up_proj`、`down_proj`。

对策略 $\pi$，聚合同一 phase/method/bucket/type 中所有被选模块的局部误差：

$$
X_{r,m,b,t}(\pi)
=
\sum_{i:\,r_i=r,\,m_i=m,\,b_i=b,\,t_i=t}
e_{r,b,t,m}.
$$

这里的 layer bucket 不是把8层假设成具有相同权重，而是用较低维度表达浅层、中层和
深层的系统性差异。逐层独立拟合会引入过多自由参数，而完全忽略层位置又无法表达深度
敏感性。

训练前使用训练集的平均 feature mass 归一化：

$$
c
=
\frac{1}{|\mathcal D_{\mathrm{train}}|}
\sum_{\pi\in\mathcal D_{\mathrm{train}}}
\sum_{r,m,b,t}X_{r,m,b,t}(\pi),
\qquad
\widetilde X(\pi)=\frac{X(\pi)}{c}.
$$

### 2.4 三档精度模型

消融采用逐步叠加的结构。

#### Q0：Local sum

所有局部误差共享一个非负全局系数：

$$
\widehat y_{\mathrm{local}}(\pi)
=
\beta
+
\operatorname{ReLU}(g)
\sum_{r,m,b,t}\widetilde X_{r,m,b,t}(\pi).
$$

该模型验证“局部输出误差简单累加”是否已经足够。

#### Q1：Structure-aware

在 Q0 上加入 method、layer bucket 和 module type 的可分解校准项：

$$
w_{m,b,t}
=
\operatorname{ReLU}\!\left(g+u_m+v_b+z_t\right),
$$

$$
\widehat y_{\mathrm{struct}}(\pi)
=
\beta
+
\sum_{r,m,b,t}
\widetilde X_{r,m,b,t}(\pi)w_{m,b,t}.
$$

该结构允许模型自然区分量化与稀疏化、不同深度位置以及不同 Linear 类型，又避免为
每个四维格点单独拟合参数。

#### Q2：Full phase-aware

最终模型进一步加入 phase 校准项：

$$
w_{r,m,b,t}
=
\operatorname{ReLU}\!\left(g+p_r+u_m+v_b+z_t\right),
$$

$$
\widehat y_{\mathrm{full}}(\pi)
=
\beta
+
\sum_{r,m,b,t}
\widetilde X_{r,m,b,t}(\pi)w_{r,m,b,t}.
$$

ReLU 保证每个局部扰动的预测贡献非负，避免模型通过不合理的正负抵消拟合训练集。
它也比早期 softplus 参数化允许精确的零贡献，更适合表示某些近似无害的
method/phase/type 组合。

### 2.5 拟合与数据划分

当前固定策略集包含72个覆盖不同方法、层位置、模块类型和阶段组合的受控策略：

- 54个训练策略；
- 18个严格 holdout 策略；
- holdout 由确定性的 feature-space farthest-point coverage 选择；
- 划分过程不读取 NLL 标签；
- dense BF16 和四个 uniform anchor 固定保留在训练集。

模型使用 float64、Adam、学习率0.03训练4000步。训练目标为：

$$
\mathcal L
=
\frac{1}{|\mathcal D_{\mathrm{train}}|}
\sum_{\pi\in\mathcal D_{\mathrm{train}}}
\left(\widehat y(\pi)-y(\pi)\right)^2
+
0.05\lVert\theta\rVert_2^2,
$$

其中偏置 $\beta$ 不参加正则化。

### 2.6 精度模型消融结果

| 累积设计 | Holdout MAE | Holdout RMSE | Holdout Spearman |
|---|---:|---:|---:|
| Q0: Local sum | 0.0571 | 0.0711 | 0.920 |
| Q1: + Structure-aware | 0.0468 | 0.0568 | 0.944 |
| Q2: + Phase-aware | **0.0413** | **0.0529** | **0.963** |

相对于 Q0，完整模型将 holdout MAE 降低27.8%。更重要的是 Spearman 从0.920
提高到0.963，说明模型不仅改善绝对误差，也更适合约束优化中的候选排序。

### 2.7 Calibration policy 数量敏感度

065进一步固定同一组18个holdout，在不改变模型结构和训练超参数的情况下，将原始54个
训练policy与新增18个calibration-only policy合并为统一的72-policy候选池。每个规模均从
该池进行10组固定随机嵌套抽样，并始终保留5个dense/uniform anchor；这样避免了新增策略
只在54之后出现造成的分布分段。论文实际使用的原始54-policy模型保持冻结，并作为独立
星标展示，不能与随机池在 $n=54$ 时的median混为一谈。

随机候选池的median holdout MAE在45/54/60/64/66个策略时分别为0.0403/0.0408/
0.0392/0.0394/0.0387，表现为较缓的平台区；使用完整72-policy候选池时进一步降到
0.0340。预先定义的
“进入72-policy完整模型MAE的5%范围且后续保持在范围内”规则仍将72识别为经验拐点。
冻结主模型的MAE为0.0413、成本为48.6 GPU-minutes；完整池成本为71.4 GPU-minutes。
因此扩展实验验证了模型可以从更多结构覆盖中继续获益，但不支持把54宣称为渐近统计
拐点；主实验只将其表述为冻结的预算/精度工作点。

论文主图将校准预算显示到64个策略，用于聚焦实际预算区间；完整72-policy审计曲线和
全部数值仍保留在065中，不能将截断图解释为已经验证64之后完全饱和。

完整结果和曲线位于
`artifacts/debug/065_llama31_quality_calibration_size_sensitivity/`。

## 3. 速度建模

速度模型分成两个层级：

1. **Kernel latency model**：预测单个 Linear module 所使用 kernel 的时间；
2. **E2E composition model**：将所有 module/phase 时间组合并映射到真实 vLLM 请求时间。

这两个层级分别报告误差，不能把二者的 MAPE 当成同一组测试。

### 3.1 Kernel benchmark 与方法集合

对每类 kernel 收集多个 $(M,N,K)$ shape 的 module-forward latency，重复测量后以
median 作为标签。模型分别覆盖：

- dense BF16；
- dense NVFP4 W4A4；
- sparse BF16 2:4；
- sparse NVFP4 W4A4；
- Marlin NVFP4 W4A16。

每类 kernel 具有独立的 hard shape support。无法满足 backend 整除约束的动作直接从
求解器动作空间移除，而不是给出虚假的 latency 或静默回退。

### 3.2 Profile-calibrated Roofline baseline

对 kernel $k$ 和 shape $s=(M,N,K)$，定义有效计算工作量 $F_k(s)$ 和显存流量
$D_k(s)$。从训练 benchmark 中估计：

$$
P_k^{\mathrm{eff}}
=Q_{0.99}\!\left(\frac{F_k(s_j)}{t_j}\right),
\qquad
B_k^{\mathrm{eff}}
=Q_{0.99}\!\left(\frac{D_k(s_j)}{t_j}\right).
$$

因此底层不是直接使用 GPU datasheet 理论峰值，而是使用当前 GPU、CUDA、CUTLASS
和 wrapper stack 上的实测有效吞吐与带宽。基础预测为：

$$
\ell_k^{\mathrm{roof}}(s)
=
\max\!\left(
\frac{F_k(s)}{P_k^{\mathrm{eff}}},
\frac{D_k(s)}{B_k^{\mathrm{eff}}}
\right)
+t_k^{\mathrm{launch}}.
$$

量化和稀疏 kernel 还使用按 $K$ 或 $N$ 插值得到的实测低分位 latency floor，以
表示 launch、packing、decompression 和特定维度造成的下界：

$$
\ell_k^{\mathrm{base}}(s)
=
\max\!\left(
\ell_k^{\mathrm{roof}}(s),
f_k^{K}(K),
f_k^{N}(N)
\right).
$$

不是所有 kernel 都同时启用两个 floor；实现根据各 backend 的测量特征选择。

### 3.3 Method-aware work 与 traffic

| Kernel | 有效工作量 | 主要 traffic approximation |
|---|---:|---|
| dense BF16 | $2MNK$ | $2MK+2NK+2MN$ |
| sparse BF16 | $MNK$ | $2MK+2MN+2NK/1.777778$ |
| dense NVFP4 | $2MNK$ | $2MK+0.5MK+0.5NK+(MK+NK)/16+2MN$ |
| sparse NVFP4 | $MNK$ | $2MK+0.5MK+0.25NK+(MK+NK)/32+2MN$ |
| Marlin W4A16 | $2MNK$ | $2MK+0.5NK+NK/16+2MN$ |

dense/sparse NVFP4 的 activation traffic 显式包含在线 activation quantization 与
packing；Marlin W4A16 保留 BF16 activation，因此不包含 activation packing。这也是
不能仅用纯 GEMM FLOPs 预测 module latency 的原因。

### 3.4 Shape-dependent residual correction

Roofline 保留了物理外推能力，但无法完全表达 tile utilization、shape tail、kernel
scheduling 和 packing efficiency。为此模型学习实测 latency 相对 Roofline baseline
的对数 residual：

$$
y_j^{\mathrm{res}}
=
\log\!\left(
\frac{t_j}{\ell_k^{\mathrm{base}}(s_j)}
\right).
$$

shape features 包括：

- $\log_2 M$、$\log_2 N$、$\log_2 K$ 及其乘积和比例；
- arithmetic intensity；
- tile 数量和总 tile 数；
- $M/N/K$ tail fraction；
- output size、weight reuse；
- activation、weight 和 scale bytes；
- small-$M$、large-$M$、small-$N/K$ 指示特征。

标准化 feature 记为 $z(s)$。模型同时计算 ridge 外推和 $q=12$ 邻居的局部插值：

$$
\widehat y_{\mathrm{ridge}}(s)
=
[1,z(s)]^\top\theta,
$$

$$
\widehat y_{\mathrm{local}}(s)
=
\frac{\sum_{j\in\mathcal N_q(s)}
\dfrac{y_j^{\mathrm{res}}}{d(z(s),z(s_j))+10^{-6}}}
{\sum_{j\in\mathcal N_q(s)}
\dfrac{1}{d(z(s),z(s_j))+10^{-6}}}.
$$

使用最近邻距离自适应组合：

$$
\lambda(s)
=
\operatorname{clip}\!\left(
\frac{2.5}{2.5+d_{\min}(s)},
0.25,
0.95
\right),
$$

$$
\widehat y^{\mathrm{res}}(s)
=
\lambda(s)\widehat y_{\mathrm{local}}(s)
+
\left(1-\lambda(s)\right)\widehat y_{\mathrm{ridge}}(s),
$$

$$
\widehat\ell_k(s)
=
\ell_k^{\mathrm{base}}(s)
\exp\!\left(\widehat y^{\mathrm{res}}(s)\right).
$$

对于训练集中完全相同的 shape，production predictor 直接返回实测 median；严格泛化
消融会按完整 $(N,K)$ group 留出，因此测试集不存在 exact lookup 泄漏。

### 3.5 Phase-aware Linear latency aggregation

当前工作负载中：

$$
M_{\mathrm{prefill}}=B\times S=8\times2048=16384,
\qquad
M_{\mathrm{decode}}=B=8.
$$

输出长度为64时，第一个输出 token 由 prefill 产生，随后发生63次 decode transition。
因此策略的 raw Linear latency 为：

$$
L_{\mathrm{linear}}(\pi)
=
\sum_{i=1}^{M}
\widehat\ell_{a_i^{\mathrm p}}
\left(16384,N_i,K_i\right)
+
63
\sum_{i=1}^{M}
\widehat\ell_{a_i^{\mathrm d}}
\left(8,N_i,K_i\right).
$$

逻辑方法 `w4a16_ours` 在 latency predictor 中映射为 `marlin_nvfp4` kernel。

### 3.6 E2E runtime calibration

真实 vLLM 延迟还包含 attention、KV-cache、normalization、sampling、Python/C++ 调度、
phase 切换以及 kernel 之间的 composition overhead：

$$
L_{\mathrm{E2E}}(\pi)
=
L_{\mathrm{linear}}^{\mathrm{true}}(\pi)
+
O_{\mathrm{runtime}}(\pi).
$$

单个 dense-BF16 比例 anchor 只能表示乘性偏差，无法表示固定 overhead。冻结主实验
使用7个训练policy拟合单调映射。将训练点按raw latency排序后，PAVA求解：

$$
\min_{z_1,\ldots,z_n}\sum_{j=1}^{n}(z_j-L_{\mathrm{E2E},j})^2,
\qquad
z_1\le z_2\le\cdots\le z_n.
$$

违反单调性的相邻训练点会被合并为共同均值；预测时在相邻校准节点之间插值，并在
训练范围外使用边界值。该映射保持raw latency的非递减顺序，与057主实验完全一致。
正仿射映射保留为内部外推诊断，不作为主表结果。最终论文Pareto点的横轴仍然是真实
vLLM延迟。

## 4. 速度消融

### 4.1 Kernel-level 消融

为了公平验证 Roofline 本身，direct fitting baseline 与 residual model 使用相同的
shape features、标准化、ridge 强度、12-neighbor mixture、随机种子和测试集。唯一
区别是 direct fitting 直接预测 $\log t$，而我们的方法预测 Roofline 上的 residual。

测试对每类 kernel 使用三个固定随机种子，每次完整留出25%的 $(N,K)$ groups，合计
4657个测试样本：

| 累积设计 | Kernel MAPE | p90 APE | Spearman |
|---|---:|---:|---:|
| S0: Direct shape fitting | 14.80% | 37.47% | 0.957 |
| S1: + Calibrated Roofline prior | 10.78% | 27.92% | 0.962 |
| S2: + Shape residual correction | **8.01%** | **22.03%** | **0.972** |

该结果表明 Roofline prior 将 MAPE 从14.80%降到10.78%，residual correction 进一步
降到8.01%。因此 Roofline 和 shape correction 都提供独立的正收益。

在15个独立 Llama3 module profile 上排除 exact lookup 后，Roofline 与
shape-corrected 模型的 MAPE 分别为19.99%和8.29%，进一步验证 residual correction
对目标模型 shape 有效。

### 4.2 Policy-level E2E 消融

E2E calibration 使用固定7个训练 policy，并在5个未参与拟合的 heterogeneous policy
上评价：

| 累积设计 | E2E MAE | E2E MAPE | Spearman |
|---|---:|---:|---:|
| Refined-kernel raw aggregation | 499.16 ms | 18.17% | 0.300 |
| + Monotonic E2E calibration (PAVA) | **224.24 ms** | **8.30%** | **0.000** |

两行均固定使用完整的 Roofline+residual kernel predictor。第一行只聚合各phase/module
的预测时间并使用单个BF16 anchor；第二行在相同raw aggregation上加入PAVA单调runtime
calibration。因此18.17%到8.30%的改善只衡量E2E composition calibration，不再与
kernel-level shape correction混在同一条消融链中。

### 4.3 Kernel与E2E误差不可直接比较

两个层级都报告MAPE，但评价对象不同：

| Evaluation scope | 精确 MAPE | 测试对象 |
|---|---:|---|
| Kernel latency | 8.00919% | 4657个 unseen-$(N,K)$ kernel 样本 |
| Policy E2E latency | 8.30197% | 5个 held-out heterogeneous policy |

Kernel MAPE 衡量单个 module 的预测，E2E MAPE 衡量完整 vLLM 请求。论文表中应当使用
分组行明确区分，不能把二者连成同一个误差序列。

## 5. 消融组织方式

本实验采用 **cumulative ablation**，即在同一层级逐步加入设计，而不是每次回到 base
只加入一个独立组件：

$$
\text{Quality: Local sum}
\rightarrow
+\text{Structure-aware}
\rightarrow
+\text{Phase-aware},
$$

$$
\text{Kernel: Direct fit}
\rightarrow
+\text{Roofline prior}
\rightarrow
+\text{Shape correction},
$$

$$
\text{E2E: Raw aggregation}
\rightarrow
+\text{Runtime calibration}.
$$

不推荐把三个速度层级强行写成一条 MAPE 链，因为 Kernel 与 E2E 的测试对象不同。推荐
在一张表中使用三个分组：Kernel latency、E2E composition、Quality prediction。

## 6. 推荐论文表格

| Scope | Variant | Error | Rank correlation |
|---|---|---:|---:|
| Kernel latency | Direct shape fitting | 14.80% MAPE | 0.957 |
|  | + Calibrated Roofline | 10.78% | 0.962 |
|  | + Shape residual correction | **8.01%** | **0.972** |
| E2E composition | Full kernel model + single BF16 scaling | 18.17% MAPE | 0.300 |
|  | + Positive-affine calibration | **8.01%** | **0.300** |
| Quality prediction | Local sum | 0.0571 MAE | 0.920 |
|  | + Structure-aware | 0.0468 | 0.944 |
|  | + Phase-aware | **0.0413** | **0.963** |

E2E 只有5个 holdout policy，统计强度低于 Kernel 的4657个样本和 Quality 的18个
holdout。因此论文中应把 Kernel 结果作为速度模型组件消融的主要证据，把 E2E 结果
表述为 composition calibration validation，而不是夸大微小的8.15%到8.01%差异。

## 7. 可直接用于论文的文字

### 7.1 Method：质量代理

> We estimate the quality degradation of a heterogeneous policy using a
> phase- and structure-aware surrogate. For each compression method, layer
> bucket, fused Linear type, and execution phase, we first measure the local
> output relative MSE against dense BF16 under the same canonical runtime
> semantics. We aggregate the selected local perturbations into a compact
> policy feature tensor and fit it to real phase-vLLM WikiText $\Delta$NLL.
> The coefficient is factorized into global, phase, method, depth-bucket, and
> module-type terms and passed through ReLU, preventing physically implausible
> cancellation between compression errors while retaining substantially fewer
> parameters than a fully independent coefficient tensor.

### 7.2 Method：速度代理

> We model each backend kernel using a profile-calibrated Roofline prior with
> a learned shape residual. The prior explicitly accounts for method-specific
> compute, memory traffic, activation packing, weight representation, and
> launch overhead, while its effective throughput and bandwidth are calibrated
> from the target software/hardware stack. A hybrid ridge and nearest-neighbor
> residual model captures tile utilization and shape-dependent scheduling
> effects. We then aggregate the predicted Linear latency according to the
> prefill and decode shapes and use a positive-affine map to account for
> non-Linear and runtime-composition overhead without changing policy ordering.

### 7.3 Ablation

> The modeling ablation shows consistent gains from the proposed components.
> On strictly unseen $(N,K)$ kernel groups, direct latency fitting obtains
> 14.80% MAPE. Adding the calibrated Roofline prior reduces the error to
> 10.78%, and shape-residual correction further lowers it to 8.01%. At the
> policy level, PAVA monotonic runtime calibration reduces held-out E2E MAPE
> from 18.17% to 8.30%. For quality prediction, local-error summation,
> structure-aware calibration, and the complete phase-aware model achieve
> held-out $\Delta$NLL MAEs of 0.0571, 0.0468, and 0.0413, respectively, while
> rank correlation improves from 0.920 to 0.963.

### 7.4 表述边界

论文不应声称预测器完全替代真实测量。更准确的表述是：

> The surrogates are used to search and rank a combinatorial policy space;
> all reported Pareto points are subsequently re-evaluated with real vLLM
> latency, WikiText NLL, and downstream task metrics.

## 8. 实现与结果索引

下列路径均相对于 `artifacts/debug/064_llama31_modeling_ablation/`。

| 内容 | 路径 |
|---|---|
| 完整质量消融 | `scripts/run_quality_ablation.py` |
| Kernel unseen-shape 消融 | `scripts/run_kernel_generalization_ablation.py` |
| E2E 速度消融 | `scripts/run_speed_ablation.py` |
| 一键复现与两次哈希检查 | `scripts/run_all.py` |
| 精度预测和指标 | `quality/predictions.csv`, `quality/metrics.csv` |
| Kernel 预测和指标 | `speed/kernel_generalization_predictions.csv`, `speed/kernel_generalization_metrics.csv` |
| E2E 预测和指标 | `speed/predictions.csv`, `speed/metrics.csv` |
| 紧凑论文表 | `report/ablation_tables.md`, `report/ablation_tables.tex` |
| 图和 caption | `report/modeling_ablation*`, `report/detailed_caption.md` |
| Calibration-size sensitivity | `../065_llama31_quality_calibration_size_sensitivity/` |

复现命令：

```bash
conda run -n wja-cospaq \
  python artifacts/debug/064_llama31_modeling_ablation/scripts/run_all.py
```

该流程完全离线，不需要 GPU。`report/reproducibility.json` 保存连续两次执行的产物哈希。
