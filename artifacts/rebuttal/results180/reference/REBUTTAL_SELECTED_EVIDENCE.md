# Rebuttal实验结果：清晰版

先看三张**全部baseline＋DASO**主表：[Qwen3-8B](responses/reviewer2/qwen3_main_table/README.md) · [Llama-3.1-8B](responses/reviewer2/llama31_main_table/README.md) · [Llama-2-7B](responses/reviewer2/llama2_main_table/README.md)。它们采用同一展示方式：两个场景各有Avg与速度，HAWQ和DASO各两个代表点。

下面每节先给结论，再给能直接比较的表。正文不放运行编号和调试版本；细节与来源可展开。英文回复单独见[逐reviewer草稿](REBUTTAL_RESPONSE_DRAFT.md)。

默认速度相对BF16，越高越快；标明“相对INT8”的表以INT8为1。Prefill Avg是四项准确率均值，P+D Avg是三项生成分数摘要，两个Avg不互相比大小。

| Reviewer | 关心的问题 | 直接查看 |
| --- | --- | --- |
| R1 / AK6a | 实际质量预算、内存开销 | [质量预算](#quality-budget)、[运行开销](#runtime-cost) |
| R2 / Exi5 | 强基线、更多负载、真实混合调度 | [模型主表](#model-results)、[不同负载](#workloads)、[混合调度](#mixed) |
| R3 / LrT2 | 新架构、预测准确性与准备成本 | [模型主表](#model-results)、[建模与求解](#model-cost) |
| R4 / QboE | 简单剪枝、搜索成本与规模 | [剪枝](#pruning)、[建模与求解](#model-cost)、[其他材料](#other) |
| R5 / gkP9 | 模块接口、反约束优化与方法定位 | [接口与反约束](#interfaces-inverse)、[其他材料](#other) |

<a id="model-results"></a>
## 1. 模型主结果：补齐强基线后还有收益吗？（R2 / R3）

**结论：** Qwen和两套原Llama均存在有竞争力的质量–速度折中。下面直接列出三模型完整baseline与DASO的摘要主表；两个场景分别保留Avg和TTFT，P+D另列TPOT与E2E，完整任务分项可进入各模型主表查看。

### Qwen3-8B

| 方法 | Prefill Avg ↑ | Prefill TTFT × | P+D Avg ↑ | P+D TTFT × | P+D TPOT × | P+D E2E × |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 70.32 | 1.00 | 29.97 | 1.00 | 1.00 | 1.00 |
| Dense NVFP4 | 68.98 | 1.70 | 29.88 | 1.70 | 1.10 | 1.38 |
| W4A16 (FP4) | 70.29 | 0.99 | 30.04 | 0.99 | 1.83 | 1.23 |
| W4A16 (INT4) | 67.31 | 1.00 | 29.44 | 1.00 | 1.82 | 1.23 |
| Sparse BF16 (2:4) | 59.18 | 1.53 | 25.02 | 1.54 | 0.77 | 1.08 |
| Sparse NVFP4 (2:4) | 42.06 | 1.86 | 6.59 | 1.85 | 1.18 | 1.49 |
| INT8 (W8A8) | 69.97 | 1.86 | 29.72 | 1.96 | 1.39 | 1.66 |
| SmoothQuant INT8 | **70.58** | 1.89 | 29.73 | 1.95 | 1.39 | 1.66 |
| FP8 (W8A8) | 69.92 | 1.69 | 29.96 | 1.70 | 0.77 | 1.11 |
| SmoothQuant FP8 | 70.49 | 1.69 | 29.96 | 1.69 | 0.75 | 1.10 |
| SQ-INT8 → W4A16 | **70.58** | 1.89 | 29.92 | 1.95 | **1.91** | 1.93 |
| HAWQ-V3 (Balanced) | 70.47 | 1.32 | **30.27** | 1.41 | 1.21 | 1.32 |
| HAWQ-V3 (Max-speed) | 69.67 | **2.17** | 29.75 | **2.22** | 1.61 | 1.91 |
| **DASO (Balanced)** | **70.58** | 1.89 | 30.01 | **2.22** | 1.80 | 2.02 |
| **DASO (Max-speed)** | 69.67 | **2.17** | 29.95 | 2.19 | 1.86 | **2.03** |

[完整任务分项与HTML](responses/reviewer2/qwen3_main_table/README.md)

### Llama-3.1-8B-Instruct

| 方法 | Prefill Avg ↑ | Prefill TTFT × | P+D Avg ↑ | P+D TTFT × | P+D TPOT × | P+D E2E × |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | **70.07** | 1.00 | 14.51 | 1.00 | 1.00 | 1.00 |
| Dense NVFP4 | 66.52 | 1.78 | 12.09 | 1.78 | 1.25 | 1.50 |
| W4A16 (FP4) | 68.48 | 1.00 | 12.63 | 1.00 | 2.01 | 1.28 |
| Sparse BF16 (2:4) | 53.97 | 1.52 | 4.94 | 1.53 | 0.87 | 1.15 |
| Sparse NVFP4 (2:4) | 37.88 | 1.98 | 1.59 | 1.97 | 1.38 | 1.66 |
| INT8 (W8A8) | 69.68 | 1.89 | 14.83 | 1.93 | 1.52 | 1.73 |
| SmoothQuant INT8 | 69.72 | 1.90 | 14.39 | 1.93 | 1.52 | 1.72 |
| FP8 (W8A8) | 69.74 | 1.72 | **15.01** | 1.72 | 0.81 | 1.15 |
| SmoothQuant FP8 | 69.70 | 1.72 | 14.96 | 1.73 | 0.80 | 1.15 |
| SQ-INT8 → W4A16 | 69.72 | 1.90 | 14.60 | 1.93 | 2.05 | 1.97 |
| INT8 → W4A16 | 69.68 | 1.89 | 14.72 | 2.01 | 2.03 | 2.01 |
| FP8 → W4A16 | 69.74 | 1.72 | 14.62 | 1.73 | 2.01 | 1.84 |
| HAWQ-V3 (Balanced) | 69.90 | 1.31 | 14.40 | 1.37 | 1.28 | 1.33 |
| HAWQ-V3 (Max-speed) | 68.95 | **2.28** | 13.88 | 2.29 | 1.73 | 2.01 |
| **DASO (Balanced)** | 69.72 | 1.90 | 14.42 | 2.28 | 2.05 | 2.17 |
| **DASO (Max-speed)** | 68.95 | **2.28** | 13.87 | **2.32** | **2.08** | **2.21** |

[完整任务分项与HTML](responses/reviewer2/llama31_main_table/README.md)

### Llama-2-7B-Chat

| 方法 | Prefill Avg ↑ | Prefill TTFT × | P+D Avg ↑ | P+D TTFT × | P+D TPOT × | P+D E2E × |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 57.70 | 1.00 | 21.55 | 1.00 | 1.00 | 1.00 |
| Dense NVFP4 | 56.74 | 1.80 | 20.76 | 1.80 | 1.11 | 1.38 |
| W4A16 (FP4) | 57.16 | 0.99 | 21.45 | 0.99 | 1.58 | 1.21 |
| Sparse BF16 (2:4) | 47.72 | 1.60 | 10.92 | 1.61 | 0.86 | 1.13 |
| Sparse NVFP4 (2:4) | 33.90 | 2.00 | 7.17 | 2.00 | 1.20 | 1.51 |
| INT8 (W8A8) | 57.61 | 1.85 | 20.76 | 1.90 | 1.31 | 1.55 |
| SmoothQuant INT8 | **57.91** | 1.86 | 21.03 | 1.89 | 1.29 | 1.54 |
| FP8 (W8A8) | 57.33 | 1.70 | 21.49 | 1.71 | 0.83 | 1.13 |
| SmoothQuant FP8 | 57.62 | 1.70 | **21.63** | 1.71 | 0.83 | 1.13 |
| SQ-INT8 → W4A16 | **57.91** | 1.86 | 21.34 | 1.88 | 1.54 | 1.70 |
| INT8 → W4A16 | 57.61 | 1.85 | 20.91 | 1.95 | 1.59 | 1.76 |
| FP8 → W4A16 | 57.33 | 1.70 | 21.49 | 1.71 | 1.59 | 1.65 |
| HAWQ-V3 (Balanced) | 57.07 | 1.33 | 21.19 | 1.43 | 1.22 | 1.32 |
| HAWQ-V3 (Max-speed) | 56.70 | **2.19** | 20.88 | 2.19 | 1.50 | 1.79 |
| **DASO (Balanced)** | 57.33 | 1.70 | 21.13 | 2.07 | 1.54 | 1.77 |
| **DASO (Max-speed)** | 56.70 | **2.19** | 20.93 | **2.21** | **1.60** | **1.86** |

[完整任务分项与HTML](responses/reviewer2/llama2_main_table/README.md)

<details>
<summary>测量说明与来源</summary>

两个场景各自选择配置；Balanced/Max-speed是本表报告的代表点，不是跨所有预算的全局极值或共同实际质量预算。P+D代表点按已测质量–速度折中整理，属于描述性展示。

HAWQ为FP4/INT8/BF16后端的Hessian/ILP PTQ适配；两个工作点分别对应中等延迟目标和最小可行延迟端点。端点求解索引已与实际被测配置核对，相同Prefill配置共享同一组数值。

速度按各自campaign的BF16归一化，不是全表同轮重测。全部旧点保留在[Llama-3.1原汇总](responses/reviewer2/llama31_8b_instruct_table2_night161/README.md)和[Llama-2原汇总](responses/reviewer2/llama2_7b_chat_table2_night161/README.md)。

</details>

<a id="quality-budget"></a>
## 2. 质量预算：要求的损失是否真的满足？（R1）

**结论：** 三模型、四个预算和两种验证样本数，共24个预算设定，在确认样本上实测的**平均NLL损失**均未超限。它们复用了14个模型/配置选择，不是24次独立试验。

这里的损失用ΔNLL表示，越小越好；小于允许上限即满足本表的平均损失要求。以下展示64个验证样本时的代表结果。**这些是按NLL预算单独选择的配置，不是上方Balanced/Max-speed的精度声明。**

| 模型 | 允许损失上限 | 实测平均损失 | 平均损失满足？ |
| --- | ---: | ---: | --- |
| Qwen3-8B | 0.02 | -0.0264 | 是 |
| Qwen3-8B | 0.04 | -0.0155 | 是 |
| Llama-3.1-8B | 0.02 | 0.0086 | 是 |
| Llama-3.1-8B | 0.04 | 0.0187 | 是 |
| Llama-2-7B | 0.02 | 0.0043 | 是 |
| Llama-2-7B | 0.04 | 0.0235 | 是 |

<details>
<summary>测量说明与来源</summary>

先用一组样本验证并冻结选择，再用另一组不重叠的64个样本检查；检查结果不参与重新选点。使用公开WikiText中的新窗口，排除旧校准/验证/测试分词流中共享的连续16-token片段，不是外部新语料。

选择规则是验证均值＋2SE＋0.005；上表六个确认均值及均值＋2SE均低于预算。完整24设定中，平均损失全部满足；另加余量的确认保护值并非全部满足（21/24），所以不称为硬质量保证。

这些预算选点可能与模型主表的Balanced/Max-speed不同。NLL是新的确认结果，速度继承同一配置已有正式测量，不是同步新测；NLL预算也不保证每项下游生成分数。

[六个预算与同配置速度](response174/budget_speed.csv) · [完整计数](response174/budget_counts.csv) · [Qwen来源](responses/reviewer1/fresh_quality_run044/README.md) · [Llama-3.1来源](responses/reviewer1/fresh_quality_run045/README.md) · [Llama-2来源](responses/reviewer1/fresh_quality_run046/README.md)。

</details>

<a id="workloads"></a>
## 3. 不同负载：优势是否只依赖原配置？（R2 / R3）

**新增邻近配置结果：** [实测精度＋暂估速度总览](responses/reviewer2/qwen_original_quality_run050/PREVIEW.md)按四组核心配置展示BF16、HAWQ/DASO各两个冻结点，保留TTFT/TPOT/E2E与包含恢复的cycle。[B8/B4完整精度](responses/reviewer2/qwen_original_quality_run050/README.md)已完成；速度暂用049旧三轮中位数等待夜间复测，不能当作新的正式速度。[固定S/O NLL](responses/reviewer2/qwen_nearby_dual_scenario_run049/SUMMARY.md)与原任务Avg分列，Avg按map/batch共用，不与下方六负载历史数值混合。

**结论：** 固定Qwen主表的P+D策略，直接比较HAWQ-V3两个点、DASO两个点及简单阶段组合。DASO Balanced在6/6负载的E2E中位数快于HAWQ Max-speed，DASO Max-speed在5/6负载快于它；HAWQ通常有更低的NLL损失，两个长Decode负载则是简单阶段组合最快。

**精度为各负载实际测得的全模型ΔNLL（越低越好），不是预测值，也不是三项生成任务的Avg。** 三项任务未按这些固定负载逐一重跑，不能将主表Avg复制到这里。全部速度**相对同负载BF16**，越高越快；cycle包含下一批Prefill前恢复。

[全部九种方法、六张完整对照表](responses/reviewer2/qwen_workload_comparison/README.md) · [分组HTML](responses/reviewer2/qwen_workload_comparison/table.html) · [54行完整CSV](responses/reviewer2/qwen_workload_comparison/full_comparison.csv)

**中心行的两点速度排序尚不稳定：** Balanced三轮E2E为1035/988/990 ms，Max-speed为1072/1065/966 ms，后者有明显Decode波动。Max-speed是沿用主表的配置名称，不能解释成每个负载都最快；上述次数仅描述已有测量中位数。[逐轮核对与严格复测缺口](results177/center_speed_followup/README.md)。

| B / 输入 / 输出 | 方法 | 实测ΔNLL ±2SE ↓ | P+D TTFT × | P+D TPOT × | P+D E2E × | 完整cycle × |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 8 / 2048 / 64 | SQ-INT8 → W4A16 | 0.02009 ± 0.01376 | 1.936 | 1.597 | 1.774 | 1.587 |
| 8 / 2048 / 64 | HAWQ-V3 (Balanced) | -0.00247 ± 0.00910 | 1.408 | 1.209 | 1.310 | 1.302 |
| 8 / 2048 / 64 | HAWQ-V3 (Max-speed) | -0.02007 ± 0.01541 | 2.201 | 1.595 | 1.891 | 1.827 |
| 8 / 2048 / 64 | DASO (Balanced) | 0.00861 ± 0.01642 | 2.213 | 1.763 | 1.996 | 1.831 |
| 8 / 2048 / 64 | DASO (Max-speed) | 0.01418 ± 0.01526 | 2.164 | 1.573 | 1.864 | 1.707 |
| 1 / 2048 / 128 | SQ-INT8 → W4A16 | 0.02506 ± 0.00984 | 1.853 | 1.745 | 1.757 | 1.547 |
| 1 / 2048 / 128 | HAWQ-V3 (Balanced) | 0.00446 ± 0.00641 | 1.430 | 1.215 | 1.233 | 1.223 |
| 1 / 2048 / 128 | HAWQ-V3 (Max-speed) | -0.01621 ± 0.00995 | 2.674 | 1.637 | 1.703 | 1.621 |
| 1 / 2048 / 128 | DASO (Balanced) | 0.02153 ± 0.01253 | 2.668 | 1.851 | 1.910 | 1.712 |
| 1 / 2048 / 128 | DASO (Max-speed) | 0.01754 ± 0.01005 | 2.618 | 1.985 | 2.034 | 1.809 |
| 8 / 512 / 128 | SQ-INT8 → W4A16 | 0.03169 ± 0.01221 | 2.235 | 1.651 | 1.721 | 1.544 |
| 8 / 512 / 128 | HAWQ-V3 (Balanced) | 0.00471 ± 0.00783 | 1.489 | 1.200 | 1.238 | 1.229 |
| 8 / 512 / 128 | HAWQ-V3 (Max-speed) | -0.00918 ± 0.01314 | 2.750 | 1.643 | 1.754 | 1.678 |
| 8 / 512 / 128 | DASO (Balanced) | 0.02944 ± 0.01268 | 2.731 | 1.916 | 2.012 | 1.836 |
| 8 / 512 / 128 | DASO (Max-speed) | 0.02485 ± 0.01289 | 2.683 | 1.993 | 2.078 | 1.874 |
| 8 / 4096 / 64 | SQ-INT8 → W4A16 | 0.02080 ± 0.01121 | 1.857 | 1.678 | 1.802 | 1.687 |
| 8 / 4096 / 64 | HAWQ-V3 (Balanced) | 0.01414 ± 0.00906 | 1.373 | 1.160 | 1.307 | 1.301 |
| 8 / 4096 / 64 | HAWQ-V3 (Max-speed) | -0.00150 ± 0.01215 | 2.077 | 1.462 | 1.853 | 1.802 |
| 8 / 4096 / 64 | DASO (Balanced) | 0.01408 ± 0.01169 | 2.090 | 1.634 | 1.940 | 1.843 |
| 8 / 4096 / 64 | DASO (Max-speed) | 0.01842 ± 0.01049 | 2.059 | 1.677 | 1.936 | 1.830 |
| 8 / 512 / 256 | SQ-INT8 → W4A16 | 0.03075 ± 0.00903 | 2.311 | 2.020 | 2.042 | 1.902 |
| 8 / 512 / 256 | HAWQ-V3 (Balanced) | 0.00516 ± 0.00630 | 1.503 | 1.246 | 1.262 | 1.260 |
| 8 / 512 / 256 | HAWQ-V3 (Max-speed) | -0.01017 ± 0.00959 | 2.751 | 1.695 | 1.754 | 1.722 |
| 8 / 512 / 256 | DASO (Balanced) | 0.02744 ± 0.01050 | 2.742 | 1.941 | 1.991 | 1.893 |
| 8 / 512 / 256 | DASO (Max-speed) | 0.02374 ± 0.00969 | 2.666 | 1.894 | 1.942 | 1.827 |
| 1 / 512 / 256 | SQ-INT8 → W4A16 | 0.03023 ± 0.00900 | 1.050 | 1.966 | 1.939 | 1.790 |
| 1 / 512 / 256 | HAWQ-V3 (Balanced) | 0.00148 ± 0.00591 | 1.107 | 1.247 | 1.244 | 1.241 |
| 1 / 512 / 256 | HAWQ-V3 (Max-speed) | -0.00714 ± 0.01111 | 1.150 | 1.697 | 1.687 | 1.651 |
| 1 / 512 / 256 | DASO (Balanced) | 0.02641 ± 0.01106 | 1.151 | 1.940 | 1.921 | 1.813 |
| 1 / 512 / 256 | DASO (Max-speed) | 0.02445 ± 0.01014 | 0.949 | 1.922 | 1.891 | 1.761 |

<details>
<summary>测量说明与来源</summary>

所有六个预定负载、固定配置均展示，没有逐行重新调参；Balanced/Max-speed是与主表相同的配置名称，不表示每个新负载都最快。完整九方法中，DASO两个点组成的速度上包络在4/6负载最快，另2/6是简单阶段组合；不能据此宣称质量和速度全面支配HAWQ。

同轮实测：每个策略顺序测六负载，三轮中位数，BF16首尾锚定；精度按实际B/S/O在64个WikiText窗口上使用真实KV路径逐token teacher forcing，逐窗口减BF16后计算均值与SE。窗口此前已研究，跨负载有共享，不是新外部盲测。±2SE是窗口统计，图上横线为三轮最小/最大值，不作显著性声明或硬预算保证。

质量与速度均取同一负载测量轮次，不混用模型主表另一轮的中心速度。两个轮次数值代码完全相同，但输入token、Graph捕获尺寸、引擎容量和测量顺序不同，不能当作严格复测或将另一轮更有利的中心数值替入。旧测量内部的Decode波动原因尚未定位，本次未新增GPU测量。

[全部54行与协议](responses/reviewer2/qwen_workload_comparison/README.md) · [质量–速度PDF图](responses/reviewer2/qwen_workload_comparison/quality_speed.pdf) · [SVG](responses/reviewer2/qwen_workload_comparison/quality_speed.svg) · [逐块精度/三轮速度重算来源](responses/reviewer2/qwen_workload_comparison/provenance.json)。

</details>

<a id="mixed"></a>
## 4. 混合调度：同一forward有Prefill和Decode还能用吗？（R2）

**结论：** 新增实现已验证同一次forward中的逐行阶段路由。Qwen Balanced在标准负载达到1.35× BF16，Max-speed在异构长Decode负载达到1.21×。

这是单独的混合调度实验，下面在该实验的九种配置内比较；速度与模型主表分开读取。

| 方法 | 标准负载 E2E×BF16 | 异构长Decode E2E×BF16 |
| --- | ---: | ---: |
| BF16 | 1.00 | 1.00 |
| INT8 | 0.94 | 0.67 |
| SmoothQuant INT8 | 0.96 | 0.67 |
| FP8 | 1.04 | 0.78 |
| SQ-INT8 → W4A16 | 1.29 | 1.20 |
| HAWQ-V3 (Balanced) | 0.93 | 0.78 |
| HAWQ-V3 (Max-speed) | 1.01 | 0.73 |
| DASO (Balanced) | 1.35 | 1.01 |
| DASO (Max-speed) | 1.22 | 1.21 |

<details>
<summary>测量说明与来源</summary>

vLLM V1，chunked prefill预算1024；同一forward内按token行gather/apply/scatter，Prefill和Decode分别使用对应权重。真实mixed计数、各阶段token总数、BF16 no-op一致性及权重一致性均已核验。

使用两份阶段权重常驻的eager变体，代价见下一节内存表。两种场景均为8请求closed-loop burst：标准输入2048/输出64，异构prompt/输出256。它不是生产HTTP/open-loop SLA证明，也不等于主表单arena的Graph路径。

[全部九种配置、实际路由和质量](responses/reviewer2/qwen_updated_mixed_run047/README.md)。

</details>

<a id="runtime-cost"></a>
## 5. 运行开销：内存和切换代价多大？（R1 / R5）

### 5.1 两份阶段表示常驻时的内存

**结论：** Qwen Balanced的GPU峰值比BF16低约20%，同时明确付出约8.24 GiB的Host阶段权重缓存。INT8更省GPU内存，DASO提供的是另一种质量–速度–内存折中。

| 方法 | GPU Torch峰值 GiB | Host阶段权重缓存 GiB |
| --- | ---: | ---: |
| BF16 | 21.38 | 0.00 |
| INT8 | 14.99 | 0.00 |
| DASO (Balanced) | 17.10 | 8.24 |
| DASO (Max-speed) | 17.31 | 8.76 |

<details>
<summary>测量说明与来源</summary>

与上一节相同的Qwen混合调度变体、KV6 GiB。GPU峰值是Torch分配峰值，包含KV/workspace等；Host列只是阶段adapter权重缓存，不是总进程RSS。Host/GPU两项不是可以相加的同一内存口径。

[原始内存字段、峰值与协议](responses/reviewer2/qwen_updated_mixed_run047/README.md) · [原主表三模型内存](responses/reviewer1/runtime_memory_night151/README.md)。

</details>

### 5.2 把下一批恢复也计入后的速度

**结论：** Qwen两个DASO点计入恢复后，完整cycle仍约为INT8的1.11×。单请求E2E与连续批次的代价分列，避免把恢复成本漏掉。

| 方法 | E2E ×INT8 | 完整cycle ×INT8 | 恢复耗时 ms |
| --- | ---: | ---: | ---: |
| BF16 | 0.60 | 0.60 | 0.2 |
| INT8 | 1.00 | 1.00 | 0.2 |
| SQ-INT8 → W4A16 | 1.16 | 1.04 | 124.1 |
| HAWQ-V3 (Max-speed) | 1.14 | 1.10 | 45.8 |
| DASO (Balanced) | 1.21 | 1.11 | 90.0 |
| DASO (Max-speed) | 1.22 | 1.11 | 102.0 |

<details>
<summary>测量说明与来源</summary>

本表使用主表共同Graph/arena路径的同轮P+D测量，不使用上一节双驻留变体的数据。B8/输入2048/输出64；E2E已包含Prefill→Decode，cycle再计下一批之前的恢复。不能用E2E优势直接代替cycle优势。

[完整重复与恢复数据](responses/reviewer2/qwen_full_repeat_run048/README.md)。

</details>

<a id="model-cost"></a>
## 6. 建模与搜索：预测准不准，选策略花多久？（R3 / R4）

### 6.1 未见过的矩阵shape：原Hybrid预测

**结论：** 原Hybrid在shape留出和整组投影维度留出上的MAPE为4.18%–11.83%。MAPE越低，预测越接近实测。

| 后端 | 未见shape MAPE ↓ | 未见投影维度组 MAPE ↓ |
| --- | ---: | ---: |
| Dense BF16 | 7.47% | 7.17% |
| Dense NVFP4 | 8.97% | 10.91% |
| Sparse BF16 | 4.22% | 4.18% |
| Sparse NVFP4 | 10.02% | 11.83% |
| W4A16 (FP4) | 6.51% | 6.35% |

<details>
<summary>测量说明与来源</summary>

用历史完整profiling数据重新划分，并仅在各自训练分区拟合；N/K组留出比单独shape留出更严格。这是原设备/协议上的泛化重放，不是新的跨硬件实验。

[全部留出样本、误差与拟合成本](runs/004/REPORT.md)。

</details>

### 6.2 已有硬件画像：整模型部署成本估计

**结论：** 对已有画像的策略，加上事先固定的BF16非Linear残差，三个模型两场景的E2E MAPE约1.00%–1.71%。它与上一张未知shape预测表是不同验证。

| 模型 | Prefill MAPE ↓ | P+D MAPE ↓ |
| --- | ---: | ---: |
| Qwen3-8B | 1.00% | 1.05% |
| Llama-3.1-8B | 1.31% | 1.71% |
| Llama-2-7B | 1.71% | 1.12% |

<details>
<summary>测量说明与来源</summary>

各Linear画像已经包含内部激活量化、scale/metadata处理和GEMM；attention、KV、norm/activation及引擎等非Linear工作，用分配前固定的BF16残差近似覆盖，不假设为零。更换负载不能直接外推相同残差，更换硬件需要重新画像/校准。

Qwen覆盖43个已测策略场景；Llama-3.1覆盖31个核心场景，Llama-2覆盖29个，未测点不混入误差。阶段恢复另报cycle。

[三模型逐点预测与PDF/SVG](responses/reviewer3/modeling_night151/README.md)。

</details>

### 6.3 已校准后，重新选一次策略

**结论：** 两步精确代理求解合计约0.2–2.0秒，修改预算或偏好时可复用已有画像与质量模型。这里测的是**求解本身**。

| 模型 | 一次两步求解耗时范围 |
| --- | --- |
| Qwen3-8B | 0.221–1.025 s |
| Llama-3.1-8B | 0.192–1.986 s |
| Llama-2-7B | 0.196–0.494 s |

### 6.4 新增8bit后的准备增量

复用原完整策略标签时，两套Llama的增量准备如下；从零标签采集、导出与验证仍需另计，不把上面的几秒求解当作全部部署成本。

| 阶段 | Llama-3.1耗时 s | Llama-2耗时 s |
| --- | ---: | ---: |
| 新增8bit局部特征 | 98.34 | 90.77 |
| 80行Linear硬件画像 | 38.19 | 28.62 |
| 两场景质量模型拟合 | 8.31 | 8.51 |

<details>
<summary>测量说明与来源</summary>

求解时间为两次HiGHS内部wall time：先最小化预测延迟，再在同速条件下最小化预测质量损失；优化的是固定加和代理。准备表为函数内部计时。进程壁钟之和、并行总墙钟和GPU-active时间是不同口径，不混加。

[完整新点开销](responses/reviewer3/new_point_overheads_night161/snapshots/20261027_184651_final/README.md) · [标签/Hessian/导出/验证等完整成本](night151/COST_REPORT.md)。

</details>

<a id="pruning"></a>
## 7. 剪枝候选：SparseGPT比简单剪枝更合适吗？（R4）

**结论：** 同2:4结构、同Sparse BF16 kernel下，SparseGPT在MMLU和WikiText PPL上优于Wanda与Magnitude，支持它作为DASO的稀疏候选。相近速度主要来自共享kernel。

| 方法 | MMLU ↑ | WikiText PPL ↓ | Prefill ×BF16 | P+D E2E ×BF16 |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 68.44 | 9.43 | 1.00 | 1.00 |
| SparseGPT 2:4 | 42.92 | 21.44 | 1.52 | 1.15 |
| Wanda 2:4 | 37.02 | 38.27 | 1.53 | 1.13 |
| Magnitude 2:4 | 28.56 | 516.33 | 1.53 | 1.13 |

<details>
<summary>测量说明与来源</summary>

Llama-3.1-8B-Instruct的完整任务评测。PPL来自完整lm-eval word perplexity，不是少量NLL窗口取exp。Wanda采用层序传播，按原记录重建128×2048训练样本；原SparseGPT tensor未保留，因此不声称字节完全一致。

这里只对表列指标下结论；完整生成任务中Wanda在部分指标更高，全部原始结果保留。[完整质量、速度和剪枝准备成本](responses/reviewer4/pruning_llama31_run020/README.md)。

</details>

<a id="matched-allocation"></a>
## 8. 分配方法：同候选、同约束下质量代理有用吗？（R1 / R2 / R3）

**结论：** 在Llama-3.1的一个相同延迟上限对照中，DASO在近似相同实测延迟下得到更低的平均NLL损失。候选集合和求解器相同，区别是质量代理。

| 方法 | 实测平均NLL损失 ↓ | 实测E2E ms ↓ | 共同E2E上限 ms |
| --- | ---: | ---: | ---: |
| DASO | 0.00345 | 1471.71 | 1580.10 |
| Hessian / ILP | 0.00794 | 1476.07 | 1580.10 |

<details>
<summary>测量说明与来源</summary>

双方使用BF16/INT8/NVFP4相同候选和MILP，P+D阶段独立。比较对象为signed-Hessian适配；不把该单点结论推广为胜过所有Hessian变体。

NLL复用原冻结64个窗口，速度同轮补齐。DASO−Hessian的paired差值为−0.00449±0.00361（2SE），属于已研究配置的描述性结果。

[全部同候选对照与配对误差](responses/reviewer2/llama31_matched_allocator_run041/README.md)。

</details>

<a id="interfaces-inverse"></a>
## 9. 反约束与接口：能满足延迟上限、连接不同格式吗？（R5）

### 9.1 反约束与实测延迟检查

**结论：** 把问题改成“延迟不超过上限时，让质量损失最小”可以执行；已有固定代理一致性和真实运行测量支持。

| 检查内容 | 检查范围 | 实际结果 |
| --- | --- | --- |
| 正向/反向求解一致性 | 8个固定代理问题 | 8/8通过 |
| 原绝对延迟上限 | 18套配置，每套24次计时 | 0次观察超时 |
| 首轮校验后检查后两轮 | 36个目标设定，复用17套配置 | 36个均可行，后两轮0次观察超时 |

<details>
<summary>测量说明与来源</summary>

正反求解检查只针对固定加和代理，不能替代真实质量评估。18套配置包含DASO及Hessian，覆盖三个延迟比例和Prefill/P+D。

36个目标设定来自另一个预先冻结规则的分割分析：round0最大样本×1.05作筛选，round1/2每目标共16个timed观察。E2E和cycle分别定义cap；原对应点在这些cap下也无超时，所以不宣称校验降低了本批违约。分析在三轮收集完成后执行，但规则先冻结，选择先于后两轮数据读取封存；不是在线干预，也不是生产硬SLA保证。

[正反代理检查](responses/reviewer5/solver_grid_run023/README.md) · [原绝对上限实测](responses/reviewer2/llama31_matched_allocator_run041/README.md) · [预注册分割校验](responses/reviewer5/latency_validation_night161/README.md)。

</details>

### 9.2 不同压缩后端之间的模块接口

**结论：** 已测768条Linear边界均为BF16输入/输出，内部压缩权重格式不会直接变成下一模块必须接收的activation格式。

| 配置 | 实测边界条数 | 输入 | 输出 |
| --- | ---: | --- | --- |
| INT8/FP4阶段混合配置 | 256 | BF16 | BF16 |
| SmoothQuant INT8 | 256 | BF16 | BF16 |
| FP8 | 256 | BF16 | BF16 |

<details>
<summary>测量说明与来源</summary>

Llama-3.1三套配置，各128 Linear×2阶段；短输入真实forward记录dtype/shape/stride。内部激活量化、scale和metadata已计入Linear画像；阶段权重搬运与恢复另报，不能由接口dtype一致推断总转换开销为零。

[768条实际记录与接口审计](responses/reviewer5/interfaces_run025/README.md) · [连续MLP方法对检查](runs/004/results/mlp_interfaces.csv)。

</details>

<a id="other"></a>
## 10. 其他已备妥材料与仍需限定的范围

- **Figure2乱码：** 已有无字体依赖的[替换PDF](runs/008/results/figure2_outlined.pdf)和[SVG](runs/008/results/figure2_outlined.svg)，保留原公式含义。
- **相同输入如何测：** 从BF16参考模型捕获同一个模块输入，分别调用各候选；候选输出不替换参考主干继续传播的activation。[具体说明](responses/reviewer5/overnight_run008/IDENTICAL_INPUTS.md)。
- **NAS方法定位：** 聚焦固定LLM的跨后端、模块、阶段联合选择和实际部署；承认资源约束搜索与硬件预测已有先例。[定位草稿](runs/008/NAS_CONTEXT.md)。
- **更大模型规模：** 已有112个Qwen3-32B维度的真实kernel forward，支持更大矩阵的后端可执行性；尚无完整30/32B模型质量/E2E结果。[规模测量](runs/004/REPORT.md)。

完整30/32B部署及生产open-loop在线SLA仍需另补证据；当前的kernel测量和mixed burst均按各自范围使用。

[精简英文回复](REBUTTAL_RESPONSE_DRAFT.md) · [本次来源与检查](results176/audit.json) · [重排前的完整旧版](results176/reference/REBUTTAL_SELECTED_EVIDENCE.md)
