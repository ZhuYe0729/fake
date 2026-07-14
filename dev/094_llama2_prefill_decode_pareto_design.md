# Llama2-7B-chat 分阶段异构压缩 Pareto 设计说明

> 本文档记录截至 2026-07-13 的实际实现、实验口径和推荐论文表述。它刻意区分：
>
> - **已进入当前结果的组件**：WikiText pooled NLL 质量代理、kernel latency surrogate、离散 DP、vLLM 实测闭环；
> - **已验证但尚未完全回灌进求解器的组件**：policy-level 单调 E2E 速度校正与显存可行性预测；
> - **不应写成最终方法的尝试**：下游任务直接拟合、phase-separated NLL 回归、phase-local error 特征。

本文以 Llama-2-7B-Chat、vLLM phase-heterogeneous backend、`prefill-decode` 场景为主；`prefill-only` 是同一框架去掉 decode 项后的特例。

## 1. 设计目标与问题定义

给定一个已训练的 causal LM、给定推理工作负载，以及一个可供每层/模块选择的压缩 kernel 集合，目标是在**不实际枚举所有层异构策略**的前提下，求解速度与模型质量之间的离散 Pareto 边界。

对于每个 module $i \in \{1,\ldots,M\}$，选择一个分阶段动作 $a_i$：

$$a_i=(a_i^p,a_i^d),$$

其中 $a_i^p$ 用于 prefill，$a_i^d$ 用于 decoding。策略为 $\pi=(a_1,\ldots,a_M)$。我们希望在质量预算 $\epsilon$ 下最小化端到端延迟：

$$
\begin{aligned}
\min_{\pi}\quad & \widehat L_{\mathrm{e2e}}(\pi;s)\\
\mathrm{s.t.}\quad & \widehat {\Delta \mathrm{NLL}}(\pi;s)\le\epsilon,\\
& \pi\in\mathcal A_{\mathrm{runtime}},\\
& \pi\in\mathcal A_{\mathrm{memory}}.
\end{aligned}
$$

这里 $s$ 表示部署场景，$\mathcal A_{\mathrm{runtime}}$ 是 backend kernel/phase 切换的合法动作集合，$\mathcal A_{\mathrm{memory}}$ 是显存可行集合。扫过不同 $\epsilon$ 得到预测前沿；之后只测量少量非支配候选并以实测结果报告最终前沿。

本工作不将“预测器输出”直接作为论文最终纵轴或横轴：横轴是实测 E2E，质量的主纵轴是实测 WikiText $\Delta$NLL 或真实下游任务分数。

## 2. 固定实验/部署场景

当前 Llama2-7B-chat 的两个场景为：

| 场景 | batch | input length | output length | 质量目标 |
|---|---:|---:|---:|---|
| prefill-only | 8 | 2048 | 0 | prompt token NLL |
| prefill-decode | 16 | 2048 | 80 | prefill NLL + 80 倍 decode NLL |

对 prefill-decode，kernel surrogate 使用 prefill GEMM 的 $m=B\times S=32768$，decode GEMM 的 $m=B=16$，并将 decode 成本重复 $T=80$ 次。正式 vLLM 测速使用同一个 phase-heterogeneous backend，`max_model_len=2128`、`max_num_seqs=16`、`max_num_batched_tokens=32768`、BF16 activation、关闭 prefix cache，并采用 `gpu_memory_utilization=0.85`。

`.85` 并非任意调参：`.9` 下某些 NVFP4 prefill 策略会因 KV cache 预留与 NVFP4 packing/GEMM workspace 峰值竞争而 OOM；保持 workload 不变、把 KV-cache headroom 调整到 `.85` 后这些策略可运行。因此，所有应在同一张最终图中比较的 ours 点与 uniform baseline 均应使用 `.85` 口径，不能混入旧 `.8` 或 `.9` 数字。

## 3. 模块化与压缩动作空间

### 3.1 Fused module 表示

Llama2-7B 的 32 个 Transformer block 被表示为每层四个可路由的 fused module，共 $M=128$：

| fused type | 原始线性层 | $(N,K)$ |
|---|---|---|
| `qkv_proj` | q/k/v projection | $(12288,4096)$ |
| `o_proj` | attention output projection | $(4096,4096)$ |
| `gate_up_proj` | gate/up projection | $(22016,4096)$ |
| `down_proj` | MLP down projection | $(4096,11008)$ |

将 q/k/v 与 gate/up 融合，使求解变量与 backend 实际启动的 fused GEMM 一致；避免把逻辑线性层的损失或时间简单相加后再误当成 runtime。

### 3.2 方法集合及 runtime 映射

策略动作的逻辑方法集合为

$$\mathcal M=\{\text{dense BF16},\text{dense NVFP4},\text{sparse BF16},\text{sparse NVFP4},\text{W4A16-ours}\}.$$

其中 `W4A16-ours` 在质量建模时与 dense-NVFP4 共用其量化权重误差，在运行时映射为 Marlin NVFP4/W4A16 kernel。dense-NVFP4 使用 CUTLASS backend。kernel predictor 评估的 kernel 集合是 dense BF16、dense NVFP4、sparse BF16、sparse NVFP4、Marlin NVFP4。

并非所有 $(a_i^p,a_i^d)$ 都允许。当前 prefill-decode 的合法 pair 为：同一方法，或 dense-NVFP4 与 W4A16-ours 的交叉 pair；decode 中排除 sparse-NVFP4，因为给定 $m=16$、fused shape 时不存在可用 kernel。该约束是 backend capability constraint，不是由学习得到。

## 4. 精度/质量代理：local error + global calibrated aggregation

### 4.1 为什么不直接用下游任务拟合

下游生成任务（CNN/DM、DialogSum、IWSLT）昂贵、带有采样/提示词/评价指标噪声，且其分数对微小压缩差别的分辨率不稳定。它们适合**迁移验证**，不适合为大量候选策略拟合主代理。

主质量模型因此使用固定 WikiText-2 teacher-forced NLL。WikiText 既不依赖目标下游数据集，也能同时覆盖长 prefill 与 decode 分布。下游三任务只在候选前沿形成后评测，用于检查预测到的质量 trade-off 是否迁移到真实生成指标。

### 4.2 监督标签

从固定的 WikiText-2 blocks 中，对 dense BF16 与每个压缩策略使用相同 token、teacher forcing 计算：

$$
\Delta \mathrm{NLL}_{p}(\pi)=\mathrm{NLL}^{\pi}_{p}-\mathrm{NLL}^{\mathrm{dense}}_{p},
$$

$$
\Delta \mathrm{NLL}_{d}(\pi)=\mathrm{NLL}^{\pi}_{d}-\mathrm{NLL}^{\mathrm{dense}}_{d}.
$$

prefill-decode 的 pooled target 为

$$y(\pi)=\Delta \mathrm{NLL}_{p}(\pi)+T\,\Delta \mathrm{NLL}_{d}(\pi),\qquad T=80.$$

这不是任意加权：它近似一个请求中 2048 个 prompt token 后、80 个 decode token 的总增量负对数似然。实现中使用固定 100 个 2048-token blocks（每策略 204,700 prefill token 与约 8,000 decode prediction token）；初始 300-block 设计因成本高而未作为最终训练预算。

### 4.3 局部误差基础项

从离线 layer/module sensitivity 表中读取压缩方法在模块输出上的相对 MSE，记为 $e_{\ell,t,m}$：

- $\ell$：layer；
- $t$：fused type；
- $m$：逻辑压缩方法；
- dense BF16 的误差定义为 0；
- W4A16-ours 使用 dense-NVFP4 对应的 local error。

对 qkv/gate-up，分别对其组成原始线性层的 local relative MSE 取均值，得到 fused module error。该 local error 不是最终 NLL 的直接相加，而是低维全局校准模型的输入。

### 4.4 Layer bucket 化和特征构造

32 层被划为 4 个连续 bucket，每个 bucket 8 层：

$$b(\ell)=\lfloor \ell/8\rfloor.$$

对 phase $r\in\{p,d\}$，策略的张量特征为

$$X^r_{m,b,t}(\pi)=\sum_{i:\,m_i^r=m,\,b(i)=b,\,t(i)=t} e_{\ell(i),t(i),m}.$$

维度为 $5\times4\times4=80$，但参数不为每一格独立拟合。对 prefill-decode，最终选中的输入是 pooled feature：

$$X(\pi)=X^p(\pi)+80X^d(\pi).$$

关键点：**local** 体现在 $e_{\ell,t,m}$，**global** 体现在所有策略共享的 method/bucket/type 校准因子；bucket 化保留粗粒度层敏感性并抑制逐层模型的过拟合。

### 4.5 正性、归一化和全局校准模型

令训练集平均总 feature mass 为

$$c=\operatorname{mean}_{\pi\in\mathcal D_{train}}\sum_{m,b,t}X_{m,b,t}(\pi).$$

使用 $\widetilde X=X/c$。系数采用可分解的正参数：

$$
w_{m,b,t}=\operatorname{softplus}(g+u_m+v_b+z_t),
$$

$$
\widehat y(\pi)=\beta+\sum_{m,b,t}\widetilde X_{m,b,t}(\pi)w_{m,b,t}.
$$

以 Adam 最小化训练 NLL MSE，并对 $g,u,v,z$ 施加 $L_2=0.05$。正性防止“压缩误差相互抵消”这种不合理解；global/method/bucket/type 分解大幅降低自由度，归一化避免 raw tensor size 或模块数量主导损失。

优化器仅需要策略之间的增量质量，因此在 DP 中省略全局截距 $\beta$；这不改变任意两个候选的质量排序或约束可行性。

### 4.6 校准集、验证集和最终模型选择

最终质量代理来自 72 个固定、受控、层异构的 calibration policies：54 个固定训练策略、18 个从未参与拟合的 holdout 策略。训练与 holdout 不是“72 个求解出的 Pareto 点”，而是专门设计来覆盖层位置、模块类型、方法和 prefill/decode 异构组合的监督策略。

最终采用 `normalized_pooled`，而不是下列替代：

- **phase-separated regression**：分别拟合 prefill/decode 再作 $p+80d$ 合成；在 decode 上尺度不稳定、过拟合；
- **phase-local output error**：额外收集 phase/method local MSE，复杂度更高但 holdout 排序下降；
- **下游任务分数拟合**：只做迁移评测，不用于主拟合。

关键 holdout 指标如下：

| 场景 | 最终代理 | holdout MAE | holdout RMSE | holdout Spearman |
|---|---|---:|---:|---:|
| prefill-only | normalized pooled | 0.126 | 0.126 | 0.774 |
| prefill-decode | normalized pooled | 0.738 | 0.818 | 0.774 |

这些指标证明排序能力足以做候选筛选，不代表质量可被无误差预测。因此论文中应把模型表述为 **surrogate-guided search**，而不是严格质量约束保证；最终曲线必须以实测 NLL/任务分数进行非支配筛选。

## 5. 速度代理：kernel surrogate 与 E2E 校正

### 5.1 对“roofline model”的精确表述

你的理解是正确的：底层 `KernelLatencyPredictor` 的每个 GEMM kernel model 都是**calibrated roofline-style baseline + shape residual correction**。此前“不是纯解析 roofline”的表述不是否定 roofline，而是强调它不使用 GPU datasheet 的理论峰值，也不止使用一个 $\max(\mathrm{compute},\mathrm{memory})$ 公式。

统一形式为：

$$
\widehat \ell_k(m,n,k)
=\ell^{\mathrm{base}}_k(m,n,k;\theta_k^{\mathrm{roof}})
\cdot r_k(m,n,k;\theta_k^{\mathrm{res}}),
$$

其中 $\ell^{\mathrm{base}}$ 是具有 FLOP、bytes、带宽和 launch-floor 物理含义的 calibrated roofline baseline，$r_k$ 是由实测数据学习的 shape-dependent residual factor。对于 benchmark 中已出现的 exact $(m,n,k)$，模型直接返回实测 median latency，而不是 residual 外推。

所以论文推荐称为：**a kernel-specific, profile-calibrated roofline model with local shape-residual correction**。这比“纯 roofline model”准确，也比“黑盒 latency regressor”更能体现设计。

预测器包含五类 kernel 的独立 latency model，并暴露 NVFP4 backend materialization/conversion model：

- `dense_bf16`；
- `dense_nvfp4`；
- `sparse_bf16`；
- `sparse_nvfp4`；
- `marlin_nvfp4`；
- `canonical_to_cutlass` 和 `canonical_to_marlin` conversion cost。

每个 kernel 先施加 hard shape support 约束，再预测 latency；不支持的动作直接从优化器动作集合移除，而不是回退为 dense。输出还保留 training-range、nearest shape、feature distance、confidence 等 routing diagnostics。

### 5.2 物理 roofline baseline 的校准方式

以 dense BF16 为例，对 GEMM $(m,n,k)$：

$$F=2mnk,\qquad D_{\mathrm{BF16}}=2(mk+kn+mn).$$

从该 kernel 的 benchmark rows 计算每个样本的 achieved throughput：

$$P_j=F_j/t_j,\qquad W_j=D_j/t_j.$$

有效峰值不是 RTX 5090 的理论参数，而是本 CUDA/kernel/software stack 的实测 99th percentile：

$$P_{\mathrm{eff}}=Q_{0.99}(P_j),\qquad W_{\mathrm{eff}}=Q_{0.99}(W_j).$$

再令 $\ell_0=\min_j t_j$ 为实测 module-forward launch floor，得到：

$$
\ell^{\mathrm{base}}_{\mathrm{dense~BF16}}
=\max\left(\frac{F}{P_{\mathrm{eff}}},\frac{D_{\mathrm{BF16}}}{W_{\mathrm{eff}}}\right)+\ell_0.
$$

因而“calibration”至少有两层含义：

1. **kernel-profile calibration**：从 benchmark 而非 datasheet 标定有效计算峰值、有效带宽与 launch floor；
2. **shape-residual calibration**：从 benchmark 学习实际 latency 相对于上述 baseline 的比例。

这种方式既保留 compute-bound/memory-bound 的物理归因，也能吸收实际 kernel 的 tile、packing、scheduler 等偏差。

### 5.3 各 kernel 的 roofline work/traffic 建模

不同 backend 的有效 work 和 traffic 不同，不能共用 dense BF16 的 bytes 公式。

| kernel | effective work | 关键 bytes / 额外项 | hard support |
|---|---|---|---|
| dense BF16 | $2mnk$ | $2(mk+kn+mn)$ | 无额外整除约束 |
| sparse BF16 (2:4) | $mnk$ | BF16 activation/output + $2nk/1.777778$ sparse weight | $M\bmod8=0,N\bmod8=0,K\bmod64=0$ |
| dense NVFP4 (W4A4) | $2mnk$ | BF16 activation read、FP4 activation packing、FP4 weight、scale、BF16 output | $N\bmod32=0,K\bmod32=0$ |
| sparse NVFP4 (2:4 W4A4) | $mnk$ | BF16+FP4 activation、$0.25nk$ sparse FP4 weight、scales、output | $M\bmod32=0,K\bmod64=0$ |
| Marlin NVFP4 (W4A16) | $2mnk$ | BF16 activation、$0.5nk$ FP4 weight、weight scale、output；**无 activation packing** | $N\bmod64=0,K\bmod128=0$ |

更明确地，dense NVFP4 的 traffic approximation 是：

$$
D_{\mathrm{dense~NVFP4}}
=\underbrace{2mk}_{\text{BF16 activation read}}
+\underbrace{0.5mk}_{\text{FP4 activation packing}}
+\underbrace{0.5nk}_{\text{packed FP4 weight}}
+\underbrace{(mk+nk)/16}_{\text{scales}}
+\underbrace{2mn}_{\text{BF16 output}}.
$$

Marlin W4A16 则为：

$$
D_{\mathrm{Marlin}}
=2mk+0.5nk+nk/16+2mn.
$$

这解释了 phase heterogeneity 的核心机会：prefill 是大 $m$，W4A4 的 activation packing 可被高 work amortize；decode 是小 $m$，W4A16/Marlin 避免 activation quantization 的固定开销，常有更合适的 latency profile。

对 sparse BF16，baseline 还包含按 $K$ 的插值 floor；对两个 NVFP4 W4A4 model，还包含 $K$ 和 $N$ floor。这些 floor 来自不同 $K/N$ benchmark group 的低分位实测 latency，用来表征稀疏解压、scale path、算法切换和小矩阵固定成本，防止纯 work/bytes roofline 对长 $K$ 或特殊 $(N,K)$ 系统性乐观。

所有量化/稀疏 kernel 的一般 baseline 形式可写为：

$$
\ell^{\mathrm{base}}_k
=\max\left(\frac{F^{\mathrm{eff}}_k}{P^{\mathrm{eff}}_k},
\frac{D_k}{W^{\mathrm{eff}}_k}\right)
+\ell^{\mathrm{floor}}_k,
$$

并在有 shape floor 时取 $\max(\ell^{\mathrm{base}}_k,\ell_{K/N\text{-floor}})$。

### 5.4 Shape residual：为什么仍需要学习项

roofline baseline 无法完整描述 tile occupancy、tail tile、small-$M$ decode、kernel scheduling、quantization/packing、sparse metadata/decompression 与 fixed $(N,K)$ 算法选择。因此各 kernel 都学习：

$$r_k(m,n,k)=\exp(f_k(\phi(m,n,k))).$$

特征 $\phi$ 包括：

- $\log_2 m,\log_2 n,\log_2 k$，以及 $\log_2(mn),\log_2(mk),\log_2(nk),\log_2(mnk)$；
- arithmetic intensity 和 shape aspect ratios；
- decode/small-$M$ 与 prefill/large-$M$ indicator；
- 128×128×64 tile count proxy 与 $m/n/k$ tail fraction；
- output size、weight reuse；
- 量化路径额外的 activation/weight/scale traffic，稀疏路径额外的 sparse weight bytes。

训练时首先以 $y_j=\log(t_j/\ell^{\mathrm{base}}_j)$ 为标签训练 ridge residual anchor；预测时在标准化 feature space 找最近的 benchmark shapes，以 inverse-distance 加权 local residual，并与 ridge residual 混合：

$$
\log r_k=\alpha(d)\log r_{\mathrm{local}}+(1-\alpha(d))\log r_{\mathrm{ridge}},
$$

其中 $d$ 是最近训练 shape 的 feature distance，距离近时更依赖局部实测，距离远时更多依赖可外推的 ridge anchor。dense BF16 residual 被下界截断为 1，保证未测 shape 不会预测得比 calibrated roofline 下界更快。

这就是“roofline + calibration”的第三层：不是单一全局乘数，而是**global physical calibration + local shape calibration + residual regularization**。

### 5.5 从 kernel 模型到 phase-policy latency

#### 原始 phase-policy latency 合成

对 module $i$、类型 $t_i$、阶段动作 $(m_i^p,m_i^d)$：

$$
\widehat L_{\mathrm{raw}}(\pi)=
\sum_i\left[
\widehat k_{m_i^p}(BS,N_i,K_i)+
T\widehat k_{m_i^d}(B,N_i,K_i)+
\widehat c_i(m_i^p,m_i^d)
\right].
$$

$\widehat c_i$ 是所需 CUTLASS/Marlin cache materialization 的预测成本。在代码中，只要该 module 的任一 phase 使用相应 NVFP4 backend，便把相应 conversion candidate 加入该动作成本。这个项是 policy/runtime materialization 的工程近似，不能误称为常规 GEMM FLOP 时间。

原始 policy sum 是 kernel-level roofline/residual model 在 graph level 的可加近似。它非常适合比较大量 module action，但没有包含：vLLM scheduler、KV cache、phase state materialization、Python/engine launch、kernel overlap、内存峰值和不同 kernel 的 runtime interaction。因此它只应作为**搜索级 base model**。

### 5.6 Policy-level 单调 E2E 校正

在 035 debug 实验中，对多个可行策略按正式 runner 测得 $L_{\mathrm{e2e}}$，学习单调映射：

$$\widehat L_{\mathrm{e2e}}=h(\widehat L_{\mathrm{raw}}),$$

其中 $h$ 由按 raw latency 排序的 isotonic regression（PAVA）得到，区间内线性插值、区间外截断。它的作用不是凭空制造更快策略，而是把 raw sum 系统性高估的中段加速压回现实 E2E 尺度，同时保持速度排序先验。

在 9 个可行校准点上：

| 模型 | 评估 | MAE |
|---|---|---:|
| point-0 单点缩放 raw surrogate | 校准点 | 326.0 ms |
| 单调 E2E correction | leave-one-out | 90.8 ms |

这验证了“kernel base + calibration factor/curve”的必要性。更准确的论文措辞是：**we calibrate a kernel-level latency surrogate with a monotone policy-level E2E calibrator**，而不是固定乘一个常数。当前实现中的 $h$ 是非参数单调曲线；若要写成 calibration factor，可在论文中写成 $\gamma(\widehat L_{raw})$，满足 $h(x)=\gamma(x)x$。

### 5.7 显存可行性

显存可行性不是压缩率单调函数：部分 NVFP4 prefill 策略在 `.9` 因 workspace/KV-cache headroom 冲突 OOM，而更快的全 W4A16 endpoint 反而可行；在 `.85` 下原 OOM 点可运行。因此当前可行性处理为：

1. kernel predictor 排除不支持的 shape/kernel；
2. 对候选做轻量 vLLM feasibility probe；
3. OOM 明确记录为不可行或调整部署的 KV-cache headroom；
4. 正式同图结果固定 `.85` 后重新全量测量。

静态 memory predictor 尚未完成。论文中不能声称已有学习式 OOM classifier；应称 runtime feasibility filtering / profiling-based feasibility check。

### 5.8 当前实现状态：重要边界

034 的首次 DP 使用的是 $\widehat L_{raw}$，035 的 $h$ 在之后被验证并用于指导重新选择/补测候选。换言之，**已完成主曲线的候选生成并非完全由 calibrated $\widehat L_{e2e}$ 重求得到**。最终论文版本应把 $h$ 和 feasibility filter 放到 solve loop 内；当前 036 的中间点补全正是该闭环的过渡实现。

## 6. 约束优化与离散求解

### 6.1 Multiple-choice knapsack 形式

每个 fused module 是一个 group，group 内为所有合法 phase-action。对 action $a$ 预先计算质量成本 $q_{i,a}$ 与时间成本 $\ell_{i,a}$。在预算 $\epsilon$ 下：

$$
\min_{a_i\in\mathcal A_i}\sum_i\ell_{i,a_i}
\quad\text{s.t.}\quad
\sum_iq_{i,a_i}\le\epsilon.
$$

这是 multiple-choice knapsack 的最短时间变体。由于质量代理为正性可加模型，$q_{i,a}\ge0$；dense BF16 action 的增量质量为 0，保证每个 group 有可行基线。

### 6.2 Budget 离散化 DP

将最大预测质量 cost 划分为 $K$ 个 bins（当前主 solver 为 1600；dense-grid debug 可使用 4000）。对动作质量：

$$b_{i,a}=\begin{cases}
0,&q_{i,a}=0,\\
\max(1,\lceil q_{i,a}\cdot K/Q_{max}\rceil),&q_{i,a}>0.
\end{cases}$$

DP state $D_j[u]$ 存储完成前 $j$ 个 module、累计 bin 为 $u$ 时的最小 latency 和 backpointer：

$$D_j[u]=\min_{a\in\mathcal A_j,~b_{j,a}\le u}\{D_{j-1}[u-b_{j,a}]+\ell_{j,a}\}.$$

每轮保留同一或更低质量 bin 下 latency 更低的 state，删除被支配 state。随后对一组质量预算求解，并对 module-level assignment 去重；若两个 budget 输出相同 policy，只保留一个。

### 6.3 Budget sweep 与端点

当前初始 sweep 使用从 0 到 $10^0$ 的对数间隔 quality ratio：dense endpoint 对应 0，最大预算端点对应近似贪心 max-speed。prefill-decode 输出了 12 个唯一策略。增大 budget grid 只会提供更多 candidate，不保证实测 E2E 连续；036 的实验表明最后若多个同 shape `o_proj` 同时翻转，仍会出现离散速度跳变。

为补齐这类跳变，可对相邻 policy 的 differencing modules 做 module-level interpolation：从低质量损失 endpoint 开始，按已知增量动作逐个/分组打开，生成额外候选，再用实测 E2E 选择。这是对离散 module action space 的必要 refinement，不是修改质量模型。

### 6.4 策略物化

DP 输出每个 fused module 的 `{prefill_method, decode_method}`，写为 phase-hetero JSON。导出器将原始 Llama checkpoint 转换/缓存为可加载 checkpoint，并保存 manifest。vLLM backend 在 prefill 与 decoding 边界按 policy 切换可用 backend。

## 7. 搜索、验证与报告闭环

推荐的最终工作流如下：

1. **离线构建 local error 和 kernel model**：不触碰目标下游数据；
2. **WikiText calibration**：生成固定 heterogeneous policies，固定 train/holdout，采集 pooled NLL，训练质量代理；
3. **E2E calibration**：从 raw solver 的稀疏覆盖点中选择策略，测量正式 vLLM E2E，拟合单调 $h$；
4. **校正求解**：在 DP cost 中使用 $h$（或能等价分解/近似的 policy-level correction），并结合 runtime feasibility filter；
5. **自适应补点**：对预测曲线中的速度空档增加 target-speed 或相邻-policy interpolation candidates；
6. **正式实测筛选**：对候选进行重复 E2E 测量和 100-block WikiText NLL；按实测两维 non-dominance 选择最终 frontier；
7. **真实生成任务验证**：只在保留点上测试 CNN/DM、DialogSum、IWSLT；最终图的纵轴为实际 ROUGE-L/SacreBLEU，baseline 与 ours 的速度按同一部署协议比较。

当前 035 已完成第 1、2、3、6、7 步的大部分，036 正在做第 5 步的 gap refinement。第 4 步（将 calibrated E2E 和 feasibility 显式整合进 DP）是论文正式算法实现前应补齐的关键工程项。

## 8. 实测口径与避免误读

### 8.1 速度

最终速度应报告 E2E median 和 speedup：

$$\mathrm{speedup}(\pi)=L_{\mathrm{dense~BF16}}/L_{\pi}.$$

当前严格 stability runner 使用“每个 repeat 单独 vLLM 进程”，因此包含可重复的 fresh-process runtime，但很慢：每次都加载约 7.9 GiB checkpoint、初始化 engine 后仅执行一个请求。它适合少数正式点；大量候选应用一进程多 repeat 的 continuous phase runner 预筛，再对保留点 fresh-process 复测。

不要混淆：请求的 `elapsed_ms` 不含权重加载；慢的是获得很多独立样本时反复重启 vLLM 的 wall-clock，而非单请求推理本身。

### 8.2 质量

WikiText quality 曲线使用实测 `target_delta_nll`。下游生成质量以 CNN/DM ROUGE-L、DialogSum ROUGE-L、IWSLT SacreBLEU 报告。已有全量策略×任务结果；它们是 transfer validation，不反向训练质量模型。BERTScore 在一批恢复任务中因 CUDA fallback 不稳定未作为新的全量汇总指标，不能把它写成该批全量曲线的统一纵轴。

### 8.3 不稳定点

若一个策略重复测量跨度异常大（当前旧 point 9 为 3245--7189 ms），必须显示或单独列出，但不能加入 frontier。不能以其最好一次速度替代 median，也不能因质量良好而绕过速度稳定性要求。

## 9. 推荐论文 Design 叙事

可按以下四个模块组织：

1. **Phase-aware heterogeneous action space**：将 Transformer 切为 runtime-fused module，联合选择 prefill/decode backend；
2. **Calibrated quality surrogate**：以 local module error 为基础，用 global/method/bucket/type 正性校准拟合 WikiText pooled NLL；
3. **Hardware-aware latency surrogate**：由 kernel benchmark-trained latency predictor 组合各 phase 的 GEMM 与 backend materialization cost，并以单调 E2E mapping 校正；
4. **Constrained discrete optimizer with measurement feedback**：multiple-choice knapsack DP 生成候选，runtime feasibility filter 和真实 E2E/NLL 决定最终实测 Pareto。

建议主张：

- 我们利用**局部误差与全局校准的结合**预测跨层异构策略的质量排序；
- 我们利用**kernel-level latency model 加 policy-level E2E calibration**避免只按压缩率估速；
- 我们以**真实测量**而不是 surrogate 值绘制最终曲线；
- 分阶段异构策略在多个工作负载下提供比 uniform compression 更细的速度—质量 trade-off。

不建议主张：

- “精度预测完全准确”或“严格满足质量约束”；
- “速度模型是纯 roofline analytical model”；
- “已有完备静态 OOM predictor”；
- “所有 uniform baselines 都被 ours 严格支配”（应由每张实际图决定）；
- 将 debug 034 raw-solver 结果直接称作 calibrated-solver final result。

## 10. 关键实现与产物索引

| 组件 | 主要路径 |
|---|---|
| WikiText calibration policies/NLL/proxy | `artifacts/debug/033_llama2_7b_chat_wikitext_phase_nll_proxy/` |
| proxy fitting | `033.../scripts/fit_ablations.py` |
| raw DP solver | `artifacts/debug/034_llama2_7b_chat_wikitext_pareto_solver/scripts/solve_predicted_pareto.py` |
| `.85` E2E calibration | `artifacts/debug/035_llama2_prefill_decode_e2e_speed_model/` |
| isotonic E2E calibrator | `035.../scripts/fit_monotone_e2e_calibrator.py` |
| formal actual NLL/E2E summary | `035.../report/formal_util085_actual_nll_summary.csv` |
| all real-task results | `035.../task_quality_all/summary.csv` |
| real-task Pareto plots | `035.../task_quality_all/report/` |
| intermediate gap refinement | `artifacts/debug/036_llama2_prefill_decode_intermediate_points/` |
| kernel latency interface | `fake/kernels/cutlass/cutlass_wrapper/modeling/kernel_predictor.py` |

## 11. 最终落地前的检查清单

- [ ] 将 isotonic E2E calibration 与 runtime feasibility filter 真正纳入 solve loop；
- [ ] 对每个最终入选点做同协议、单卡、足够重复的 E2E 复测；
- [ ] 使用连续 runner 快速预筛大量中间点，避免为每个候选反复加载 vLLM；
- [ ] 以实测 WikiText NLL 重新计算 non-dominance；
- [ ] 在至少一个真实生成指标上展示与 WikiText trend 一致的 trade-off；
- [ ] 图注中写清 workload、`.85`、E2E median、样本数和 point 9 等排除规则；
- [ ] 明确 baseline 质量来自相同提示模板/数据集，baseline 速度来自同一 `.85` 部署协议。
