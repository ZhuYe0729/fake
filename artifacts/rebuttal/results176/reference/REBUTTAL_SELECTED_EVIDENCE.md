# Rebuttal精选实验证据与回复草稿（五位人工reviewer）

更新：2026-10-28 04:38 UTC。本文件供组织rebuttal使用，精选已经完成、能够支撑明确主张的结果；不搬运全部实验、探索过程或失败日志。文中每张表均链接完整来源。 可直接编辑的精简英文版本见[逐reviewer回复草稿](REBUTTAL_RESPONSE_DRAFT.md)。

**当前状态：034–048共15个后续run均已完成并通过各自审计，047 mixed与048统一重复也已完成。** 最新三模型汇总为[Qwen 34行](responses/reviewer2/qwen3_table2_night161/README.md)、[Llama-3.1 30行](responses/reviewer2/llama31_8b_instruct_table2_night161/README.md)、[Llama-2 29行](responses/reviewer2/llama2_7b_chat_table2_night161/README.md)。这些是跨campaign证据汇总；下面用于直接比较的Qwen P+D表统一采用048同轮结果。

建议主线：**补齐强基线与新架构 → 展示质量接近BF16时的真实加速 → 用新窗口实测回答预算可靠性 → 以多负载、真实mixed路由及成本数据回应系统问题。** 完整30/32B部署尚无结果，不将大shape测量当成完整大模型评测。

记号：D1指冻结的Qwen `w1_e0.01`，D32指`w32_e0.03`。二者是探索性配置，名字中的q及prefill权重描述原代理偏好；**实际质量预算统一记ε，并由单独的validation/confirmation实验判断**。下面的速度和生成质量表不自动构成ε保证。

| Reviewer | 最值得放入回复的证据 | 本文位置 |
| --- | --- | --- |
| R1 / AK6a | 新窗口实测预算、显式验证筛选、阶段表示与峰值内存 | 第1节 |
| R2 / Exi5 | INT8/FP8/SmoothQuant/HAWQ强基线，同候选控制，六负载，真实mixed forward | 第2节 |
| R3 / LrT2 | Qwen架构、两类延迟预测验证、非Linear处理与准备成本 | 第2、3、4节 |
| R4 / QboE | 简单剪枝比较、亚秒至秒级代理求解、Figure2修正版 | 第4节 |
| R5 / gkP9 | 扩展8bit动作、BF16接口实证、反约束求解、identical inputs解释 | 第2、5节 |

## 1. R1 AK6a：质量预算的实测支持与阶段表示内存

### 表1：64窗口validation选点后，新64窗口confirmation的代表预算

每个模型固定已有候选池；先用`mean + 2SE + .005`筛选并冻结选点，再开启新确认窗口。新数据排除旧校准、旧validation/test分词流共享的连续16-token片段；新validation和confirmation也互不重叠。未重新拟合代理或按确认结果修改选点。

| 模型 | 实际ε | 冻结配置 | Validation保护值 | 新确认ΔNLL ±2SE | 确认均值+2SE |
| --- | --- | --- | --- | --- | --- |
| Qwen3-8B | 0.02 | uniform_int8 | 0.00436 | -0.02637 ± 0.01414 | -0.01222 |
| Qwen3-8B | 0.04 | w1_e0.04 | 0.03924 | -0.01548 ± 0.02269 | 0.00720 |
| Llama-3.1-8B | 0.02 | uniform_sq_int8 | 0.01617 | 0.00864 ± 0.00528 | 0.01393 |
| Llama-3.1-8B | 0.04 | w1_q0.01 | 0.03467 | 0.01866 ± 0.00933 | 0.02799 |
| Llama-2-7B | 0.02 | uniform_int8 | 0.01012 | 0.00428 ± 0.00720 | 0.01148 |
| Llama-2-7B | 0.04 | w32_q0.03 | 0.03867 | 0.02353 ± 0.01161 | 0.03514 |

**可用主张：** 上述三个模型在ε=.02/.04的六个代表点，确认均值及“均值+2SE”均低于请求预算；严格预算允许回退BF16。表内确认上界是均值+2SE，区别于另加.005的筛选保护值；SE是经验统计，不是硬保证。原始q与实际ε不混用。

这些是固定有限池的新窗口复核，不是全候选空间重新搜索，也不是新语料/下游任务盲测；同一模型的多个预算可能复用同一确认配置。来源：[Qwen044](responses/reviewer1/fresh_quality_run044/README.md)、[Llama3.1 045](responses/reviewer1/fresh_quality_run045/README.md)、[Llama2 046](responses/reviewer1/fresh_quality_run046/README.md)。

**完整预算计数：** 044–046共24个预先固定预算标签（ε=.01/.02/.04/.07，validation N=32/64），实测确认均值超限0/24；标签复用了14个模型/配置选择，不是24次独立试验。这里的0/24只指**实测均值**，不意味着另加余量的确认保护值全部通过。

### 表1补充：同一预算选点对应的实际速度

以下选ε=.04、validation N=64；策略map及源policy文件SHA逐一核对相同。NLL是044–046的新确认，速度继承原032/034/035同map的正式测量，并非新确认同时重测的速度。

| 模型 | 实际ε | 配置 | 新确认ΔNLL ±2SE | 历史同map E2E ×BF16 | 速度来源 |
| --- | --- | --- | --- | --- | --- |
| Qwen3-8B | .04 | w1_e0.04 | -0.01548 ± 0.02269 | 2.061 | 032 |
| Llama-3.1-8B | .04 | w1_q0.01 | 0.01866 ± 0.00933 | 1.891 | 034 |
| Llama-2-7B | .04 | w32_q0.03 | 0.02353 ± 0.01161 | 1.770 | 035 |

Qwen的这一预算选点是`w1_e0.04`，与表3的D1/D32是不同map；因此只用于说明**NLL预算下的速度折中**，不沿用D1/D32的生成分数。NLL预算也不等于每个下游任务无损。[完整关联CSV及map SHA](response174/budget_speed.csv) · [来源审计](response174/analysis_audit.json)。

### 表2：实际mixed运行时的内存（Qwen，run047）

同GPU、KV6GiB、两份阶段表示常驻的运行变体。GPU字段按storage去重，只是adapter/Linear相关子集；Torch峰值另含KV、workspace等。host字段是额外CPU权重缓存，不与GPU峰值相加。

| 配置 | GPU adapter字段 GiB | Torch进程峰值 GiB | Host adapter缓存 GiB |
| --- | --- | --- | --- |
| BF16 | 12.938 | 21.384 | 0.000 |
| DASO D1 | 8.588 | 17.104 | 8.236 |
| DASO D32 | 8.757 | 17.306 | 8.757 |

D1的该口径GPU峰值较BF16低约**20.0%**，同时明确计入8.236GiB host阶段缓存。这个结果回应“双表示是否一定使GPU显存高于BF16”；不把此变体与主表单arena的恢复开销混算。来源：[047完整内存与调度](responses/reviewer2/qwen_updated_mixed_run047/README.md)。

**英文回复草稿：**

> We added an explicit validation-and-confirmation procedure: policies are selected using a guarded validation estimate and frozen before evaluating new, disjoint confirmation windows. Across Qwen3-8B and both Llama models, the six representative selections at ε=0.02/0.04 have measured confirmation means plus two standard errors below the requested budgets. This is empirical evidence for the validation-enhanced procedure, rather than a hard guarantee from the additive surrogate alone. We also report the representation overhead explicitly: in the mixed-runtime experiment, a representative DASO policy uses 17.104 GiB of peak Torch-allocated GPU memory versus 21.384 GiB for BF16, while requiring 8.236 GiB of host-side phase-weight storage.

## 2. R2 Exi5 / R3 LrT2：强基线、新架构与不同负载

### 表3：Qwen3-8B统一完整P+D重复（run048）

同GPU6/NUMA1、KV6GiB、B8/S2048/O64、当前共同Graph/数值路径；三轮独立进程、每轮3 warmup/5 timed、BF16首尾锚点。全部方法均完整生成CNN/DM 1000、DialogSum 1500、IWSLT 333条；native Qwen模板、关闭thinking、greedy、最多256新token。CNN/DM和DialogSum为ROUGE-L%，IWSLT为corpus BLEU。速度先按各轮BF16归一化再取均值；cycle额外包含下一批prefill前的恢复。

| 方法 | CNN/DM | DialogSum | IWSLT | E2E ×BF16 | E2E ms | Cycle ×INT8 |
| --- | --- | --- | --- | --- | --- | --- |
| BF16 | 22.824 | 26.957 | 40.132 | 1.000 | 1999.2 | 0.601 |
| INT8 W8A8 | 22.747 | 26.822 | 39.586 | 1.665 | 1201.0 | 1.000 |
| SmoothQuant INT8 | 22.696 | 27.174 | 39.322 | 1.661 | 1203.3 | 0.998 |
| FP8 W8A8 | 22.846 | 26.974 | 40.066 | 1.112 | 1797.3 | 0.668 |
| SmoothQuant FP8 | 22.842 | 27.030 | 40.013 | 1.101 | 1815.7 | 0.661 |
| 固定SQ-INT8 prefill / W4A16 decode | 22.714 | 27.062 | 39.988 | 1.934 | 1033.7 | 1.037 |
| HAWQ-FP4 PTQ，r=.80 | 22.828 | 27.354 | 40.636 | 1.319 | 1515.4 | 0.787 |
| 共同硬件最小成本端点 | 22.656 | 27.112 | 39.489 | 1.905 | 1049.5 | 1.097 |
| DASO D1（探索点） | 22.707 | 27.335 | 39.995 | 2.017 | 991.0 | 1.108 |
| DASO D32（探索点） | 22.586 | 27.359 | 39.909 | 2.032 | 983.7 | 1.106 |

**优先使用的结论：**

- D1/D32达到**2.017× / 2.032×** BF16 E2E速度，相对INT8分别为1.212× / 1.221×；计入恢复后仍为1.108× / 1.106× INT8。
- D1三项生成分数接近BF16，并在本次重复中取得较INT8更高的DialogSum与IWSLT分数。三任务等权摘要比INT8高0.294分，描述性5000次paired bootstrap区间为[0.067, 0.530]；该区间以已选配置为条件，未校正探索选择或多重比较，不称确证性显著提升。回复优先展示三个分项。
- 这组比较已经包含uniform INT8、SQ-INT8、FP8、SQ-FP8、简单阶段组合及硬件端点。HAWQ为**Hessian/ILP的PTQ硬件适配**；表中r=.80是其延迟目标，不等于DASO的质量预算。用本表说明不同的速度–质量取舍，不写成同预算击败HAWQ或所有生成指标全面最优。

来源：[048完整报告](responses/reviewer2/qwen_full_repeat_run048/README.md)、[逐项CSV](responses/reviewer2/qwen_full_repeat_run048/repeated_pd.csv)、[描述性区间](responses/reviewer2/quality_uncertainty_all_baselines_night161/qwen3/README.md)。原Llama也已补齐相同8bit/HAWQ及固定阶段控制，见[30行Llama3.1](responses/reviewer2/llama31_8b_instruct_table2_night161/README.md)与[29行Llama2](responses/reviewer2/llama2_7b_chat_table2_night161/README.md)。

### 表3补充：Llama-2原模型上的代表质量–速度折中（run035）

同campaign内的精选点，完整三生成任务；其他SQ-FP8/HAWQ和阶段控制结果见29行主表。原q是偏好预算，实际新窗口ε=.04复核见表1最后一行。

| 方法 | CNN/DM | DialogSum | IWSLT | E2E ×BF16 |
| --- | --- | --- | --- | --- |
| BF16 | 23.604 | 21.608 | 19.441 | 1.000 |
| INT8 W8A8 | 23.663 | 21.746 | 16.873 | 1.553 |
| SmoothQuant INT8 | 23.555 | 21.318 | 18.203 | 1.542 |
| FP8 W8A8 | 23.615 | 21.797 | 19.047 | 1.128 |
| 固定SQ-INT8 prefill / W4A16 decode | 23.898 | 21.655 | 18.469 | 1.696 |
| DASO wP=32探索点 | 23.838 | 21.712 | 17.851 | 1.770 |

DASO代表点达到1.770× BF16，相对同轮INT8为1.139×；计入恢复后为1.063× INT8。与INT8相比，CNN/DM和IWSLT的点估计更高，DialogSum接近；三任务摘要差异不作为显著精度提升宣称。它展示原Llama模型上可用的折中点，不声称同时支配所有阶段组合或FP8。来源：[035完整扫点](runs/035/SWEEP_REPORT.md)、[原始全精度表](runs/035/results/sweep_all_results.csv)。

### 表4：隔离候选与求解器差异的质量代理比较

Llama-3.1，P+D独立阶段、同BF16/INT8/NVFP4候选、同Linear延迟上限、同MILP，只替换质量代理。代表中等目标r=.80，原绝对E2E上限1580.10ms。NLL来自已冻结的原64窗口，速度在041同轮补齐。

| 分配器 | ΔNLL ±2SE | E2E ms | 原上限超限/24次 |
| --- | --- | --- | --- |
| DASO | 0.00345 ± 0.00228 | 1471.71 | 0/24 |
| Signed Hessian / ILP | 0.00794 ± 0.00370 | 1476.07 | 0/24 |

DASO在近似相同实测延迟下ΔNLL更低；paired差值DASO−Hessian为−.00449±.00361（2SE）。该代表点支持“收益并非只来自增加候选或更换求解器”。比较对象限定signed-Hessian适配，不外推到所有Hessian变体。来源：[041同候选对照](responses/reviewer2/llama31_matched_allocator_run041/README.md)。

### 表5：固定中心配置迁移到六种负载（run040）

没有逐负载重新优化D1/D32。报告相对同轮INT8的速度，全部六个预定负载均展示；所有参考方法另见来源。E2E包括P→D，cycle再包括恢复。

| B/S/O | D1 E2E ×INT8 | D32 E2E ×INT8 | D1 cycle ×INT8 | D32 cycle ×INT8 |
| --- | --- | --- | --- | --- |
| 8/2048/64 | 1.199 | 1.112 | 1.100 | 1.018 |
| 1/2048/128 | 1.338 | 1.391 | 1.199 | 1.237 |
| 8/512/128 | 1.317 | 1.360 | 1.202 | 1.227 |
| 8/4096/64 | 1.159 | 1.157 | 1.101 | 1.094 |
| 8/512/256 | 1.354 | 1.284 | 1.288 | 1.208 |
| 1/512/256 | 1.324 | 1.313 | 1.249 | 1.222 |

**可用主张：** 两个冻结DASO点在六种负载上，E2E及含恢复cycle均快于INT8；优势覆盖B=1/8、S=512/2048/4096、O=64/128/256，而非只依赖原B8/S2048/O64。这里是相对INT8的迁移证据，不宣称每个负载都胜过全部阶段组合。新负载质量按各自实测判断，不沿用中心q作保证。来源：[040六负载完整结果](responses/reviewer2/qwen_updated_workloads_run040/README.md)。

![六负载固定策略的速度与完整cycle](response174/qwen_six_workloads.png)

[可导出PDF](response174/qwen_six_workloads.pdf) · [SVG](response174/qwen_six_workloads.svg) · [图与全精度数据说明](response174/FIGURE.md)。点为三轮相对INT8速度的中位数，误差线为完整最小/最大范围，不是置信区间；部分cycle范围跨过1，优势表述对应中位数。

### 表6：实际chunked/mixed forward已经能运行（run047）

vLLM V1，chunked prefill预算1024；同一forward中的prefill/decode token逐行gather/apply/scatter，分别使用对应阶段权重。BF16 routed/no-op检查、真实mixed计数、阶段token总数和权重一致性均通过。此处使用dual-resident eager变体，与表3的Graph/arena路径分开报告。

| 负载 | 策略 | E2E ×BF16 | E2E ×INT8 | 实际路由ΔNLL ±2SE |
| --- | --- | --- | --- | --- |
| B8/S2048/O64 | D1 | 1.348 | 1.450 | 0.00506 ± 0.01817 |
| B8/S2048/O64 | D32 | 1.225 | 1.332 | 0.01504 ± 0.01429 |
| 异构prompt，B8/O256 | D1 | 1.013 | 1.506 | 0.02608 ± 0.01114 |
| 异构prompt，B8/O256 | D32 | 1.210 | 1.800 | 0.02122 ± 0.01119 |

**可用主张：** 阶段相关策略并非必须假设整个forward只有一个阶段；新增实现实际支持mixed token路由。D1在标准burst为1.348× BF16，D32在异构长decode burst为1.210× BF16。保留两种策略在两case的结果；不以每case最好点冒充一套统一策略。此实验是两个closed-loop burst，双驻留有额外host/GPU表示成本，不写成生产open-loop SLA证明；本表NLL窗口已在此前研究中查看过。

来源：[047协议、实际路由与完整结果](responses/reviewer2/qwen_updated_mixed_run047/README.md)。

**英文回复草稿：**

> We expanded the comparison to INT8, SmoothQuant INT8, FP8, SmoothQuant FP8, a hardware-aware Hessian/ILP PTQ adaptation, and simple uniform phase combinations. Under the same Qwen3 runtime and measurement campaign, two representative DASO policies reach 2.017×/2.032× BF16 E2E speed, compared with 1.665× for INT8; including restoration, they remain 1.108×/1.106× faster than INT8. We report individual generation metrics and the quality–speed trade-offs. Both frozen policies also outperform INT8 in E2E and full-cycle time across six batch/input/output configurations, without per-workload retuning. To address chunked prefill directly, we additionally implemented and verified token-row routing for genuinely mixed prefill/decode forwards. This dual-resident runtime achieves representative 1.348× and 1.210× BF16 speedups on two burst workloads; we distinguish it from the original arena runtime and do not equate these burst experiments with production SLA guarantees.

## 3. R3 LrT2：延迟模型准确性与非GEMM成本

### 表7：原Hybrid的广shape留出验证

历史完整profiling数据重新划分并仅在训练分区拟合；shape留出与整个N/K组留出分别测试。这是原Hybrid泛化能力的证据，区别于已知shape的实测LUT。

| 方法 | shape测试点 | shape MAPE | N/K组测试点 | N/K组 MAPE |
| --- | --- | --- | --- | --- |
| dense_bf16 | 643 | 7.47% | 652 | 7.17% |
| dense_nvfp4 | 276 | 8.97% | 258 | 10.91% |
| sparse_bf16 | 244 | 4.22% | 258 | 4.18% |
| sparse_nvfp4 | 202 | 10.02% | 214 | 11.83% |
| marlin_nvfp4 | 214 | 6.51% | 216 | 6.35% |

来源：[004广覆盖验证与数据](runs/004/REPORT.md)。这些是历史设备/协议的重放，不冒充新增跨硬件盲测。

### 表8：三模型策略级E2E预测（冻结LUT + BF16固定残差）

预测偏置在分配前固定；下表六组均来自同GPU0/NUMA0的新主表campaign，不根据正式测试修改残差。

| 模型 | 场景 | 已测速配置 | E2E MAPE |
| --- | --- | --- | --- |
| Qwen3-8B | Prefill | 16 | 0.996% |
| Qwen3-8B | P+D | 27 | 1.052% |
| Llama-3.1-8B | Prefill | 13 | 1.312% |
| Llama-3.1-8B | P+D | 18 | 1.714% |
| Llama-2-7B | Prefill | 13 | 1.707% |
| Llama-2-7B | P+D | 16 | 1.119% |

**解释方式：** 各压缩Linear的真实画像已经包含内部量化、scale/metadata及GEMM；attention、KV、norm/activation和引擎等非Linear工作以每负载BF16残差近似处理，阶段恢复另报cycle。表8支持已画像shape上的成本加和有效性，表7支持Hybrid在未见shape上的插值/泛化，二者不互相替代。更换硬件需重新画像/校准；不是完全无测量的预测。

来源：[三模型建模与可导出PDF/SVG](responses/reviewer3/modeling_night151/README.md)。

**英文回复草稿：**

> We now report two distinct validations. For the original hybrid latency estimator, held-out shape and N/K-group MAPE ranges from 4.18% to 11.83% across the five backends. Separately, the measured-LUT deployment extension obtains 1.00%–1.71% policy-level E2E MAPE across Qwen3 and both Llama models. The latter uses a BF16 residual fixed before policy allocation to approximate non-Linear work, rather than treating attention, normalization, or engine overhead as zero. Backend profiling includes activation quantization and metadata processing; phase restoration is reported separately through full-cycle latency. Hardware-specific profiling is required, and its measured preparation cost is reported separately from policy solving.

## 4. R4 QboE：简单剪枝、搜索耗时与呈现修复

### 表9：同2:4结构、同SparseBF16 kernel的简单剪枝对照

Llama-3.1-8B-Instruct，完整任务评测；Word PPL来自完整WikiText lm-eval，并非64窗口NLL取exp。速度按各自三轮BF16锚点归一化，来源campaign单列于原表。

| 方法 | MMLU % | WikiText word PPL | Prefill ×BF16 | P+D E2E ×BF16 |
| --- | --- | --- | --- | --- |
| BF16 | 68.445 | 9.425 | 1.000 | 1.000 |
| SparseGPT 2:4 | 42.921 | 21.436 | 1.520 | 1.149 |
| Wanda 2:4 | 37.018 | 38.268 | 1.525 | 1.128 |
| Magnitude 2:4 | 28.564 | 516.328 | 1.528 | 1.130 |

**可用主张：** 在相同稀疏kernel成本下，SparseGPT保留了更好的MMLU/PPL，支持其作为DASO稀疏候选的选择；DASO还可以跨量化/剪枝候选选择，不必固定全模型剪枝。这里只比较表列指标，不把它写成SparseGPT在所有生成任务都领先。来源：[020完整剪枝比较](responses/reviewer4/pruning_llama31_run020/README.md)。

### 表10：已校准后，更新预算/阶段权重的精确代理求解成本

一次先最小化延迟、再同速最小化质量损失，两次HiGHS内部wall time合计。每模块有8×8=64个合法阶段动作；仍是同一固定代理问题的求解，不是量化、导出或完整评测耗时。

| 模型 | 模块×阶段动作 | 已记录求解点 | 两次求解合计范围 |
| --- | --- | --- | --- |
| Qwen3-8B | 144 × 64 | 8 | 0.221–1.025 s |
| Llama-3.1-8B | 128 × 64 | 7 | 0.192–1.986 s |
| Llama-2-7B | 128 × 64 | 6 | 0.196–0.494 s |

Llama3.1/Llama2的新增8bit局部特征采集分别约98.34/90.77s，80行Linear画像约38.19/28.62s，两个场景质量拟合合计约8.31/8.51s。上述是函数内部增量成本，复用了已有完整策略标签；从零标签采集、导出和validation仍须计入，不能把“约0.2–2秒求解”写成全部部署准备耗时。来源：[完整新点成本](responses/reviewer3/new_point_overheads_night161/snapshots/20261027_184651_final/README.md)、[原完整成本](night151/COST_REPORT.md)。

**Figure2：** 已有消除字体依赖的[矢量PDF](runs/008/results/figure2_outlined.pdf)和[SVG](runs/008/results/figure2_outlined.svg)，保留原图与公式含义，可在修订稿中替换。来源：[008图修复审计](runs/008/REPORT.md)。

**规模补充：** [004](runs/004/REPORT.md)还完成了112个来自Qwen3-32B投影维度的真实GPU forward，覆盖四种N/K和四种M；可用于支持后端在更大矩阵上的可执行性。它使用合成权重/激活，范围限定为kernel与shape，不作为完整32B模型质量或E2E评测。

**英文回复草稿：**

> We added Wanda and magnitude pruning under the same 2:4 structure and sparse backend. SparseGPT achieves 42.92% MMLU and 21.44 WikiText word perplexity, compared with 37.02%/38.27 for Wanda and 28.56%/516.33 for magnitude pruning, while backend speed is similar. We also measured search cost explicitly: after calibration, the two-stage exact surrogate solve takes approximately 0.2–2.0 seconds across the three models. Profiling, label collection, export, and validation are accounted for separately. We additionally provide a font-independent vector version of Figure 2.

## 5. R5 gkP9：接口、8bit候选、反约束与相同输入

**动作空间扩展：** 原五候选之外已加入plain INT8、SmoothQuant INT8和FP8，形成八候选；另有固定alpha=.5的SQ-FP8完整baseline。比较见表3及两Llama最新主表。

**接口实证：** [025实测](responses/reviewer5/interfaces_run025/README.md)中，三种冻结配置、128 Linear×2阶段，共768条模块边界记录均为BF16输入/输出；此前[004的98个MLP方法对/shape组合](runs/004/results/mlp_interfaces.csv)也检查了连续模块链。因此，内部压缩权重格式不要求把压缩activation格式传播给下一层。内部量化/scale开销已在Linear画像中，阶段权重搬运与恢复另计；不能由此写成总转换开销为零。

**反约束优化：** 对固定加和代理，把延迟设为约束、质量损失设为目标，仍保留逐模块选择的最优子结构。独立[023检查](responses/reviewer5/solver_grid_run023/README.md)中8个forward/inverse Pareto一致性检查全部通过。另在[041](responses/reviewer2/llama31_matched_allocator_run041/README.md)对同候选DASO/Hessian、r=.95/.80/.65共18个固定配置补真实速度；每点24个timed观察均未越过各自原绝对E2E上限。这支持该扩展的可执行性，不把有限测量解释成生产硬SLA。

### 表11：原171规则的预注册延迟校验分析（新增CPU交付）

在041首个速度输出前已冻结源码/规则。round0的最大样本×1.05通过时才可选，round1/2各8个timed样本用于检查；E2E与cycle各自定义cap。六池×三目标×两指标共36标签、17个唯一选中map。执行分析发生在测量收集后，选择先于后两轮数据读取封存，不称实时在线干预。

| 场景/阶段 | 指标 | 标签 | 可行标签 | 后两轮选点超限 |
| --- | --- | --- | --- | --- |
| prefill_only/same_operator | E2E | 6 | 6 | 0 |
| prefill_only/same_operator | cycle | 6 | 6 | 0 |
| prefill_decode/same_operator | E2E | 6 | 6 | 0 |
| prefill_decode/same_operator | cycle | 6 | 6 | 0 |
| prefill_decode/phase_pair | E2E | 6 | 6 | 0 |
| prefill_decode/phase_pair | cycle | 6 | 6 | 0 |

**可用主张：** 在这一预先固定的分割分析中，全部36标签可行且后两轮观察0超时，支持对E2E及完整cycle分别做实测可行性校验。每标签16个观察、多个标签共用map/样本；原对应点在这些cap下也是0超时，因此不宣称校验降低了本批违约率或提供生产SLA保证。原041绝对cap未改，这里cap另按round0同场景BF16重新定义。

来源：[完整36标签与规则](responses/reviewer5/latency_validation_night161/README.md) · [汇总CSV](response174/latency_gate_summary.csv)。

**Identical inputs：** 从未压缩BF16参考模型捕获模块输入，对同一Tensor分别计算参考与各候选输出，候选输出不替换参考主干继续传播的activation；用累计SSE除以参考输出能量。局部continuation特征是teacher-forced位置合批近似，完整策略质量另外在真实vLLM KV路径评估。[实际归档实现及英文解释](responses/reviewer5/overnight_run008/IDENTICAL_INPUTS.md)。

**方法定位：** 承认hardware-aware NAS/混合精度搜索已有资源约束和预测器先例，把本文贡献聚焦于固定LLM的跨后端、模块与阶段联合选择及真实运行时部署。[已整理NAS定位草稿](runs/008/NAS_CONTEXT.md)。

**英文回复草稿：**

> Internal compressed weight formats are distinguished from inter-module activation interfaces. We recorded 768 actual Linear boundaries, all with BF16 inputs and outputs; candidate-specific activation quantization and scaling are included inside the measured Linear backend, while phase transfer and restoration are reported separately. We also implemented the inverse formulation: minimizing predicted quality loss under a latency cap preserves the additive problem's optimal substructure, and all eight forward/inverse consistency checks passed. Real measurements of 18 fixed allocation policies stayed within their original E2E caps across 24 timed observations per policy. For local perturbations, every candidate receives the same BF16-reference module-input tensor; complete-policy quality is evaluated separately through the actual runtime.

## 使用顺序与证据入口

篇幅紧时，建议优先放**表3（强基线/Qwen）＋表1（实际预算）＋表5（六负载）**；R2的mixed质疑引用表6，R3引用表7/8，R4引用表9/10，R5使用第5节。共享事实只写一次，再在各reviewer回复交叉引用。

探索点的q不写成实际ε；跨campaign汇总不写成全表同轮重测；Gen Avg不替代分项；新的mixed变体单独标注。以上保留的是影响所选正面结论成立的条件，完整研究记录仍在各run。

[原始人工reviewer意见](reference/reviewers.md) · [本文来源SHA与状态核验](selected_evidence173/source_audit.json) · [完整实验总清单](EXPERIMENT_CHECKLIST.md)
