# 十小时连续推进记录（plan151，进行中）

窗口：2026-10-26 16:00 UTC起约10小时。只处理人工reviewer。[总实验清单](EXPERIMENT_CHECKLIST.md) · [计划151](../../dev/plans/151_rebuttal_night_continuation_plan.md) · [实时监督状态](night151/status.json)。

最新完整Qwen汇总：[19行、209项指标Table 2](responses/reviewer2/qwen3_table2_run016/README.md)。两个原Llama及扩展负载继续运行，不能把此链接理解为整个夜间任务已结束。

## 18:09 UTC：Qwen三轮速度完成，Llama进入真实准备

- Qwen016的37套配置、43个场景均已完成独立64窗口确认；114次正式进程覆盖三轮同卡测速及BF16首尾锚点。[NLL/速度报告](runs/016/REPORT.md)已发布，完整任务仍在继续。
- DASO-8验证后ε=.04的P+D为2.0509×，此前强HAWQ对应的共同硬件最小成本phase-pair端点为1.8744×，同卡同轮次。此为实际测得的优势点，不能据此宣称每种预算或任务全面最优。
- 该DASO点确认ΔNLL=.007556，paired ±2SE=.021160，均值低于.04；四个验证选择点均值预算0超限。ε=.01的P+D仍回退BF16，Prefill ε=.01退化为uniform plain INT8；这些结果均保留。
- 同候选中央延迟上限下，DASO的P+D ΔNLL均值低于signed HAWQ；大多数成对±2SE区间仍跨0，不能写成已证明显著优于所有Hessian处理。signed/clip_zero/absolute的全部预定点见[配对CSV](night151/qwen016_paired_nll.csv)。
- Llama-3.1(run017)和Llama-2-Chat(run018)各自已实测768条新8bit局部误差、448条实际融合权重量化扰动，分别耗时约98.3/90.8秒；Hessian在空闲GPU5/GPU4继续，不与Qwen速度重叠。
- Llama-2原Chat权重目录为空，已恢复固定版本并核对两份权重与官方LFS SHA完全一致。两个模型的原SparseGPT状态均通过134项未剪枝norm/embedding/head身份检查；不是拿base替代chat。
- Llama-2主B8/S2048/O64需要8.25GiB BF16 KV，已在运行前固定10GiB并加入真实block容量门槛；完整生成统一batch4/10GiB。Llama-3.1保留主测/生成6GiB、生成batch8。每模型内所有方法设置相同。

## 后续已排定

1. 完成Qwen016完整任务/审计，同时监督两Llama Hessian。
2. 两Llama执行原标签数值一致性检查、新标签、八候选拟合、原策略及强基线、验证选择、确认和三轮核心速度。
3. 优先完成两个模型各14行核心主表；固定SQ-INT8-prefill/W4A16-decode简单phase对照单列。两模型核心速度优先结束，再并行完整任务，避免一模型完整任务阻塞另一模型正式测速。
4. 扩展预算、长decode/mixed、剪枝与开销等按剩余时间继续；所有未测项目明确标识。当前没有修改Qwen016冻结配置或削弱HAWQ。

这是进行中记录；完成项以对应run的完成标记、报告和审计为准，不把后台命令已启动等同于测试完成。

## 18:34 UTC：Qwen完整任务和审计完成

- run016已完成11个预登记完整任务作业，最终审计核验15个旧run的67,283项文件，封存9,900项本轮文件，17次失败命令及其日志仍保留。manifest SHA：`7366eec8710a81650875c630c85ba679b8b1116a4d426d1912e44b042286e334`。
- DASO-8 P+D ε=.04的CNN/DM、DialogSum、IWSLT分别为21.724、25.066、38.392；中心阶段适配HAWQ分别22.828、27.354、40.636。DASO此点更快，但不能宣称下游质量同时领先；NLL预算不等于跨任务保证。
- 两个Llama各80项新shape画像已完成，16项8bit数值参考门槛均通过；Llama-3.1已进入原标签缓存一致性检查，Llama-2等待独占BF16校准。
- 补齐[Qwen冻结43配置建模明细](night151/qwen016_modeling_points.csv)、[汇总](night151/qwen016_modeling_summary.csv)、[可导出图](night151/qwen016_latency_modeling.pdf)与[新八候选实际内存/恢复/捕获记录](night151/qwen016_runtime_overhead.csv)。完整点集Prefill/P+D延迟MAPE为0.996%/1.052%，最大1.652%/2.146%。这是预先测量的Linear-LUT＋固定BF16残差的结果，不是原跨shape Hybrid预测器的泛化精度；没有confirmation后拟合。
- 相同43点的质量代理MAE为.01413/.01163 nats/token，差配置与原始未验证提案均包含在内。上述不同误差不能混写成“模型整体约1%误差”。

## 18:49 UTC：两Llama多卡运行，扩展负载冻结

- 两模型各六项原标签门槛均通过：BF16、NVFP4、仅按map多样性预选的异构train配置，Prefill和P+D逐块差均为0。原144份标签可复用；新增96份标签/模型已分别在0/2/3/4和1/5/6/7卡并行。
- tmpfs容量限制并行，完整迁移五份已完成Qwen权重缓存约70.62GiB到项目XFS，并逐文件SHA验证、保留原路径链接。全部字节保留，代码/日志/结果/文档未删。[迁移记录](night151/cache_records/)保存原路径、大小、mtime和SHA。
- 自动审批曾因将项目目标误判为仅余10.7GiB的根overlay而拒绝迁移。只读核验确认目标实际在device=66305的XFS、可用约695GiB；脚本加入“不同于根overlay且总复制量外至少留32GiB”的硬门槛后获准并完成。没有绕过容量检查或写满根分区。
- [run019](runs/019/README.md)已固定11行标签/10套唯一map、六种B/S/O，包含两种O256。保留旧DASO incumbent、当前DASO-8、强HAWQ中心/端点及均匀8bit/W4A16/简单phase对照。原模型核心表完成前控制器只等待，不抢占其GPU或测速锁。
- 019为固定配置迁移，每负载新测64窗口NLL及三轮速度；不按新负载结果重新选点。沿用公开WikiText-test语料，不能写成全新未见数据。

## 20:07 UTC：原模型进入validation，补齐强端点展示

- 两模型96份新训练标签均已完成。017进入52套配置validation，018进入39套；质量代理拟合及分配正常完成，未出现失败状态。
- 在两个模型primary列表冻结及confirmation打开之前，增加共同FP4/INT8硬件最小成本端点的主表完整评测，核心表从14变15行。该端点本已在冻结pool中，不改动作、求解、预算、validation规则或数据。原三份报告/冻结/审计源码保存于各run的reference/core_before_hardware_endpoint；新增core_reporting_scope.json与其SHA绑定。
- [准备/运行时开销表](night151/COST_REPORT.md)已按叶子进程汇总，明确进程求和不是GPU活跃时间或总墙钟，缓存继承不是从零免费。确认/完整下游与准备成本分开，失败耗时保留。
- [run020 Llama-3.1剪枝控制](runs/020/README.md)代码与数据已准备，等待两个核心主表后执行。找到原SparseGPT加载器，按“非空文本双换行拼接、关闭special tokens”对齐Llama-3.1的2,435,022 token、128个原起点和2048长度。原tensor未保留，不能声称字节哈希对比；Llama-2仍有1 token差异，该可选控制没有冒充相同数据启动。

## 22:25 UTC：确认评测继续，完整MC准备复核

- 两模型校准已结束，均在confirmation。017当次检查36/52套、4张卡正常，已验证并保存17份真实核心导出缓存；剩余磁盘约330GiB。018选择点已冻结，确认与完整任务未完成前不判断最终排名。
- 五档预算[run021](runs/021/README.md)的规则/代码于20:40:47 UTC在任何source confirmation输出前冻结，共用原池和64窗口，不新增独立test。
- 完整MC在看到任何新MC输出前统一batch8，模型内所有方法一致；KV为Llama-3.1 3GiB、Llama-2 10GiB。每进程先通过B8/S2047完整prompt-logprobs门槛；scheduler预算16384确保整批prefill。生成任务及三轮正式速度设置保持原协议。
- 020的预备数据链接已复核修正，剪枝仍等待核心表完成。019/020仍按原队列，不抢占核心测试。
- 默认受限工具访问已恢复：仅为不存在的项目内只读挂载点创建空目录，无凭据读写。先前自动审批超时的记录追加现已补齐。

## 23:25 UTC 更新
- 018核心三轮速度完成，017开始三轮。两模型共8个verified预算点的confirmation均值均通过，不能将原始exact提案当成validation最终选择。
- 新[022](runs/022/README.md)仅补021预登记预算中4份非核心map的正式速度，控制器等待两份核心审计，不改变已冻结选择。
- 成本新增[精确权重缓存操作](night151/weight_cache_operations.csv)，与叶子计算命令分列，避免重复计费。

## 2026-10-27 00:35 UTC 更新
- 两Llama全部核心速度完成；完整任务8卡并行。MC门槛后的phase恢复遗漏已修复并保留首批失败，正式任务正常继续。
- [Llama同候选/预算诊断](night151/LLAMA_DIAGNOSTICS.md)覆盖12组对照与8个主预算点，未删除不利点。
- [023求解器网格/反约束复核](runs/023/REPORT.md)完成16点和8项inverse检查；已交付reviewer1/5，结论仅针对固定代理。
- [总清单](EXPERIMENT_CHECKLIST.md)已用新CSV更新三张核心表；未测质量继续明确留空。

## 2026-10-27 02:03 UTC 更新
- 两Llama共130个完整task结果（各13套×5任务）全部完成，生成仍在最后几组，所有worker正常。主表原始来源在017/018，最终15行将在生成齐全后统一发布/审计。
- [模型诊断](night151/MODELING_REPORT.md)、[18行内存/cycle](night151/RUNTIME_MEMORY_REPORT.md)、[Hessian变体](night151/HESSIAN_VARIANTS.md)和[16点solver检查](runs/023/REPORT.md)已交付。
- [024局部粒度尝试](runs/024/REPORT.md)没有整体改善，不替换方法；[025接口检查](runs/025/README.md)等源核心表后使用空闲卡。
- 019/020调度已实际启用源依赖完成后重用释放的卡，速度独占锁保留；021/022及预算发布器继续等待，未冒称完成。

## 2026-10-27 02:56 UTC 更新
- [Llama-3.1完整15行](responses/reviewer2/llama31_8b_instruct_table2/README.md)及[Llama-2完整15行](responses/reviewer2/llama2_7b_chat_table2/README.md)均已测齐/封存；连同Qwen汇总共539个指标单元完整，见[数值核对](night151/core_tables_numeric_audit.json)。
- [026 validation样本量跟进](responses/reviewer2/validation_size_run026/README.md)完成：固定14个PD incumbent，新32窗口让.04点选中1.79×端点，原32窗口选1.53×；用独立新64窗口比较均未超限，.01仍BF16。旧主表保留，速度/完整生成继承相同map；不是全候选重新优化。
- 021五预算20点均值0超限，022缺失速度仍排队；019六负载质量完成/三轮速度中；020剪枝权重和NLL完成/完整任务排队；025记录器修复后等待速度独占锁，未结束便不称完成。
