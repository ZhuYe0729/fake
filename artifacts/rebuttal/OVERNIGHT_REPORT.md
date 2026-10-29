# 五位人工reviewer补充实验汇总

本轮按plan144推进，只处理AK6a、Exi5、LrT2、QboE、gkP9。004–011八个独立实验目录均已完成可执行的预定测试、审计和封存，并按五位人工reviewer整理为21份证据包；001–003保持原封存。全部GPU作业已结束。

## 质量预算与优化：最值得优先回应的问题

[005报告](runs/005/REPORT.md)把数值网格和质量代理误差分开验证。原2000-bin DP逐模块向上取整，会将SQ-INT8的prefill预测ΔNLL从0.00180推到0.02843；精确one-hot求解在epsilon=.01下将预测Linear成本由493.78降至359.33ms。16个小实例穷举、22个前向/反向Pareto检查通过。

精确求解仍不能自动修复质量代理：本轮原DP和精确版prefill各4/4个预算点实际超限。新增独立validation后选择，8个预算点在独立test上0违约；这些预算点包含重复选中相同策略，不能说成8个独立配置的泛化证明。prefill紧预算选SQ-INT8，epsilon=.04选异构策略；P+D最紧两档回退BF16。

新增验证的代价也完整计入：38策略校准测速、38策略validation，另有37次模型导出；三GPU并行的选择前wall约38.7分钟，累计进程时间含I/O/初始化，不能叫纯GPU-active time。原标签/模型准备还需另计。该验证阶段是新增改进，不应写成原提交已有严格保证。

[010完整下游报告](runs/010/REPORT.md)补齐代表策略和简单剪枝：共36份完整任务结果，其中18份新结果、18份同权重/表示的旧结果按SHA继承；MMLU每份均为57科、14042题。

| 配置 | MMLU % | WikiText PPL |
| --- | ---: | ---: |
| BF16 | 72.874 | 13.492 |
| 验证后prefill epsilon=.01（SQ INT8） | 72.611 | 13.396 |
| 验证后prefill epsilon=.04（异构） | 71.101 | 13.892 |
| SparseGPT 2:4 | 54.102 | 20.125 |
| Wanda 2:4 | 53.561 | 23.809 |
| Magnitude 2:4 | 23.594 | 922.415 |

代表性P+D异构点完整CNN/DM1000、XSum子集1500、IWSLT333相对BF16的ROUGE-L变化分别为−0.209、−0.240个百分点，IWSLT BLEU为−0.594。生成允许EOS，不能将其generation_seconds与固定O的速度混作一个结果。WikiText NLL预算也不保证所有下游任务准确率。

## 强基线、更多负载与真实mixed调度

[003](runs/003/REPORT.md)完成16种B/S/O、267条比较、251个唯一速度组合和202份质量评测。速度统一GPU0。实际ΔNLL≤.01切片中，9个P+D负载有7个由DASO组已测点领先，两组B1由BF16领先；这是已测点的事后质量切片。原prefill重分配14/14超预算、P+D为1/18，均保留。

[009强化HAWQ报告](runs/009/REPORT.md)保留signed/clip/abs三种敏感度处理、INT4与FP4两种真实后端移植，共35比较点、16份唯一map。FP4 HAWQ是强对照：该阶段中心prefill有实际ΔNLL约.009、2.169×的点，原固定DASO无法声称全面领先。该阶段中心P+D的固定DASO为1.244×，FP4控制1.238×，差距很小。这些为009阶段记录，新增统一比较最终以011为准，不使用有利的旧时段替代后续确认。

最紧cap的部分HAWQ请求不可行，显式clamp到最小成本；P+D对应全NVFP4-prefill/W4A16-decode，收益不能全部归于Hessian精度分配。另一HAWQ map才是实际97/45/2模块的混合精度。所有移植是PTQ精度分配适配，不冒充原CNN QAT/TVM整套复现。

[006实际mixed报告](runs/006/REPORT.md)使用真实vLLM V1 chunked调度、逐行区分prefill/decode、两种真实kernel及gather/scatter，全部阶段行数和数值检查通过。它是新增双驻留变体，原global-phase单表示运行时本来不支持任意mixed forward；所有请求在burst起点到达，不是生产open-loop服务验证。

全部八策略、三种真实B/S/O均有16窗口teacher-forced质量检查，共24文件/384计分窗口/49152目标token；不同case/策略重复用同16个文本窗，不是384份独立文本。首轮BF16锚点有约6%漂移，因此追加全部八策略反向顺序重复，不挑方法/负载。两轮各27组速度，最终非BF16策略各10个timed burst、BF16各20个；初始六策略21组也完整保留。

| 实际mixed负载 | 实际ΔNLL均值上限 | uniform最快 | 固定DASO最快 | HAWQ-FP4控制最快 |
| --- | ---: | ---: | ---: | ---: |
| B8/S2048/O64，chunk1024 | .01 | 1.029× | — | — |
| 同上 | .02 | 1.029× | 1.173× | 1.149× |
| B16/S512与4096/O64，chunk4096 | .01 | 1.337× | — | — |
| 同上 | .02 | 1.337× | 1.543× | 1.444× |
| B8/S512与2048/O256，chunk1024 | .01 | 1.000× | 1.103× | 1.089× |
| 同上 | .02 | 1.000× | 1.103× | 1.136× |

这些是事后实测均值切片，质量区间/全部.01/.02/.04/.07点和逐轮速度见006，不是质量保证或生产SLA。最值得强调的是B16异长case：五候选DASO两轮均约1.54×、ΔNLL均值.01750；HAWQ实际分配点ΔNLL .01237，两轮约1.45×/1.34×。更快的全NV4/W4控制ΔNLL .03316，不在.02切片内。标准B8 case的DASO/HAWQ排名两轮反转，长decode case差距约1.3%，均不应强调显著领先。

## 011：全部强化基线和验证点的统一标准负载确认

[011完整报告](runs/011/REPORT.md)将验证后全部8点与009全部35点合并为43比较标签、19map/23唯一质量文件。质量使用同一64个B8 test窗口、同canonical/scales/backend的既有结果，逐SHA继承，不产生新标签或再次选择。速度统一GPU0、B8/S2048/O1或64；最后一轮在其他GPU实验结束后反向顺序测全矩阵，作为主结果。

| 场景 | 实际ΔNLL均值上限 | uniform最快 | 固定DASO最快 | 验证候选已测最快 | HAWQ-FP4最快 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill | .01 | 1.887× | — | 1.883× | 2.184× |
| Prefill | .02 | 1.887× | — | 2.194× | 2.184× |
| P+D | .01 | 1.000× | 1.438× | 1.369× | 1.198× |

这是已测点的事后实际质量切片，不等于部署前同epsilon规则的输出；例如表中验证P+D最快来自epsilon=.04，最紧两档规则本来回退BF16。Prefill .02的2.194×/2.184×差距很小，不声称显著领先。最终BF16锚点终点/起点为1.005（prefill）和.999（P+D）。

首轮有部分时段与006共享host，其25份原始速度文件、报告/CSV/图全部保留为diagnostic；最后一轮另有25份完整速度文件，不混合两个环境。中心P+D五候选DASO约1.247→1.438×，HAWQ全NV4/W4约1.396→1.198×，存在排名反转。当前数据不足以把时段差异归于单一原因，也不能将最终一轮的优势声称为跨时段稳定最优；006 B16异长mixed两轮约1.54×是更稳的支持点。

## 全流程开销、接口与内存

[004报告](runs/004/REPORT.md)包含36组异步/同步/双驻留控制，9组跨模式输出逐token一致；另有98个真实MLP链、12份独立profiler trace、112个真实32B维度kernel forward。

- 五种原方法广覆盖shape或N/K留出MAPE约4.2%–11.8%。极少局部训练点外推失败的负面控制保留，不能说16点足以校准新硬件。
- 003全部点的预测Linear和直接对E2E，MAPE为prefill25.38%、P+D42.94%；每负载只用BF16校准非Linear固定项后，分别为5.77%和17.36%。profile与E2E在不同5090卡，不是同卡因果拆分，也不能据此保证真实时延SLA。
- 98个MLP链确认跨模块激活接口为contiguous BF16；激活量化/打包和输出缩放在各Linear内部完成。链残差包含cache/launch等，不应称为纯格式转换开销。
- 原SparseGPT两种pattern各约305秒、HAWQ Hessian约1766秒、DASO prefill标签壁钟约4856秒；profile/拟合/搜索/导出与校准分别列出。不能只拿毫秒级搜索代表从零准备成本。
- standard负载中，五/七候选DASO双驻留Linear权重10.55/10.96GiB，低于BF16的12.94GiB，但仍有10.55/8.28GiB pinned host权重。GPU峰值还含6GiB KV及工作区，CPU RSS和pinned内存不能重复相加。
- 原异步流水线有约70–125ms恢复成本，完整cycle计入。异步不在所有策略/负载都更快，反例保留；双驻留与单表示差异也不能全部解释成DMA。

## 反向约束、简单剪枝与规模

[005反向约束](runs/005/REPORT.md)实现min预测质量损失、约束预测时延，并与前向Pareto核验。直接把Linear约束反转，实测SLA违约明显；加入预定5%校准余量后，prefill三档都选同一SQ策略，其实际5次timed测量与三档SLA的15次对照全部满足，不是15次独立计时。P+D两档被选出，其中一档仍有4/5样本违约，第三档不可行。应回应“可以反向求解”，不能回应“已获得严格真实SLA保证”。

[007简单剪枝](runs/007/REPORT.md)对magnitude、官方规则Wanda适配、SparseGPT统一普通2:4和真实cuSPARSELt sparse BF16 kernel。Wanda与SparseGPT同128个训练校准块，magnitude无校准；新增504个模块mask结构检查和官方公式/loop等价核验通过。三者prefill均约1.53×、P+D约1.09×，但质量明显不同，完整下游见010。

规模方面，完成真实Qwen3-32B的四类N/K×四种M×七方法共112项forward，以及36/64/80层动作数量的精确搜索，约0.03–0.24秒。前者合成权重、后者重复8B动作表；没有完整预训练30B/32B的质量/E2E对照，这一concern仍只部分覆盖。

## 文档与证据完整性

[008](runs/008/REPORT.md)提供原Figure2的零字体依赖[矢量PDF](runs/008/results/figure2_outlined.pdf)和SVG，已做288dpi像素对照与视觉检查。原论文PDF保持原样；不能声称本机复现了reviewer端全部乱码原因。

[NAS定位](runs/008/NAS_CONTEXT.md)承认ProxylessNAS/FBNet/BRP-NAS已有硬件预测和约束搜索；本文应突出固定LLM的异构格式/阶段配置与执行，而不夸大搜索机制原创性。[相同输入说明](runs/008/IDENTICAL_INPUTS.md)明确dense backbone相同层输入，以及原decode局部特征合批M64、非压缩KV逐步重放的近似。原拟合保持不变。

原001–003共46527份归档文件、68份共享运行时源文件及编译vLLM算子SHA检查通过。所有失败尝试、不利点、代码、日志、CSV和文档保留；只释放归档外任务拥有的可重建模型二进制，并记录cleanup。最终文件按[人工reviewer矩阵](responses/HUMAN_REVIEWER_EVIDENCE.md)和[reviewer证据目录](responses/README.md)索引；[证据包副本/链接审计](responses/overnight_packaging_audit.json)和[总封存审计](overnight_final_audit.json)记录最终来源。未向外部提交任何答复。
