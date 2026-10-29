# 144 — 人工 reviewer 核心 concern 夜间补充实验

## 授权与范围
- 用户要求003结束后继续推进其余rebuttal实验，夜间无需停下来确认；只处理reference/reviewers.md中AK6a、Exi5、LrT2、QboE、gkP9五位人工reviewer，AI Reviewer之后的内容不纳入任务。
- 目标是用有针对性、可复核的实验支持文章的真实优势。所有预定对照、不利结果和失败尝试保留；不使用测试NLL回调超参数或隐藏预算违约。
- 003保持既定工作负载/策略/质量模型并先完成统一GPU0速度、报告、审计和证据封存。准备后续脚本可并行，后续正式GPU实验在003完成后启动。
- 后续代码/数据/结果按004、005等独立编号隔离；原001/002及封存003不修改，共享vLLM/DASO源文件不修改，新增运行时能力只在实验进程中安装。

## 优先级与验收

### P0：完成003
- 16种负载、47份唯一策略、251个独立速度组合、202份质量评测，267条包含别名的比较记录。
- 全部最终速度同一GPU0，原GPU3部分完整保留为diagnostics。审计、图表、完整CSV、reviewer1/2/5证据包完成。

### P1 / run004：准备、搜索、格式与运行时开销
- 对应AK6a内存、LrT2预测准确性/准备开销/非Linear操作、QboE搜索成本、gkP9格式转换开销。
- 汇总001/002/003原始准备成本，区分串行累计GPU工作、并行壁钟、缓存复用与新负载增量成本，不能仅报DP几秒而省略标签采集。
- CPU测试36/64/80层等实际模块数量下的拟合/查询/DP或ILP时间与内存，标记为搜索复杂度扩展，不作为未部署30B模型的精度或E2E结果。
- 实际MLP gate/up→SiLU→down链测量同格式/异格式接口，核验输出/输入BF16及layout；量化、解量化属于Linear成本，链残差还含cache/launch影响，不等同于纯格式转换。
- 标准、长输入、短输入长输出三种代表负载，固定若干已冻结DASO/phase-uniform策略，对比异步拷贝、同步拷贝、双表示驻留控制；测真实TTFT/TPOT/E2E、GPU/CPU/pinned内存和传输字节，分别保留无插桩速度与profiler分解。
- 验收：所有控制语义等价（输出token/teacher-forced NLL核验）、速度同卡、profile非Linear/拷贝与适用边界可解释，完整开销表可直接用于人工reviewer3/4/5。

### P2 / run005：精确求解、反向约束与质量可行性验证
- 对应AK6a实际预算满足、gkP9严格latency SLA能否反转、QboE搜索开销。
- 在冻结原质量拟合下，对照原2000-bin DP与不量化quality的one-hot ILP，排除逐模块向上取整造成的8bit配置漏选；小实例与穷举核验，真实动作表检查uniform锚点与前/反向Pareto一致性。
- 实现min预测quality subject to预测latency预算，说明在同一可加代理下最优子结构不变；代理约束和真实TTFT/E2E SLA分别报告，不混为硬保证。
- 在测试前固定候选池、质量预算/latency预算及验证规则。训练/validation与最终test严格分离：以独立WikiText validation块做少量候选的质量/延迟验证，保留原始、精确求解、验证后选择三个版本；统一8bit/BF16锚点作为可行回退。
- 原始质量模型不因测试标签修改；验证使用预先定义的保守统计量/时延余量，正式test只评估，报告违约率、实际质量、实际速度和额外验证开销，不将经验结果写成形式化泛化保证。
- 验收：反向优化有真实选出策略及SLA审计；数值网格问题与质量代理误差分开量化；若验证改善预算可靠性，明确标为新增验证阶段，不能声称原文已具有该阶段。

### P3 / run006：真实mixed-token / chunked-prefill能力与代价
- 对应Exi5的主要拒稿理由，优先于无条件扩展模型规模。
- 审计原global-phase执行的限制；探索独立dual-resident实验变体：依据vLLM实际scheduled token位置与原prompt长度区分prefill/decode行，分别执行两种真实Linear kernel并合并BF16输出。
- 不改共享vLLM文件。先做mask/数值/phase一致性和BF16等价测试，再用真正开启chunked prefill的vLLM调度，异长请求或受限token预算确保实际mixed forward次数大于0。
- 保留uniform/no-op控制与现有phase策略，测吞吐、TTFT/TPOT/尾延迟、额外显存、gather/scatter及调度开销。原异步单表示释放实现与新增双驻留mixed实现分开报告。
- 验收：不将强制同质batch伪装成mixed serving；若工程上不能正确支持，则提交明确限制与验证，不制造生产兼容性结论。

### P4 / run007：简单结构化剪枝与规模补充
- 对应QboE简单剪枝和规模、LrT2/AK6a混合精度比较。
- 若已有证据不足，增加magnitude 2:4、Wanda 2:4对照，和SparseGPT使用同一真实sparse BF16 kernel及相同冻结数据；分别记录无校准/有校准准备成本、实际NLL/速度与选定完整任务。
- 已有模型仅Qwen3-8B、Llama系7/8B，未缓存30B。优先做有解释力的搜索规模与真实大shape kernel验证；不能用重复8B模块的CPU计时冒充30B整模型部署。只有存在可行且公平的完整baseline、足够磁盘和运行时支持时才追加真正30B实验。

### 文档与证据整理
- 人工reviewer concern→现有/新增证据矩阵，明确已解决、部分解决和限制。
- 以原始论文/官方实现核对NAS定位与Wanda公式；提供引用可追溯的答复材料。
- 检查Figure2乱码/字体嵌入；存在可恢复源时生成修正版矢量图及渲染核验，原reference PDF保留。
- 每阶段完成后追加本impl、生成独立REPORT/CSV/图/manifest和相应reviewer证据。临时模型二进制仅在全部相关评测完成后释放，代码/日志/结果CSV/文档一律保留。

## 补充P5 / run009：HAWQ敏感度处理稳健性
- 依据：冻结Hessian平均特征值252项中有35项为负；原001/003保留signed估计，可能影响异构基线的合理性。
- 在不修改001–003的前提下，固定比较signed、max(trace,0)、abs(trace)三种权重敏感度处理；分别使用原Marlin INT4/INT8/BF16绑定phase方案，以及原FP4移植的4bit prefill NVFP4/decode W4A16配对。
- 同一真实shape成本和原DASO两个预测latency cap，低于最小成本时显式记不可行并clamp；不依据最终NLL选处理方式，所有预定变体保留。只测中心prefill/P+D，避免盲目扩展。
- CPU先冻结全部策略；在空闲GPU0统一速度，质量在空闲GPU独立执行。若与优先任务争资源，优先005/006/007。五位人工reviewer对强异构基线的concern是本补充的依据。

## 补充P6 / run010：代表策略的完整下游质量确认
- 在005选择完成、尚未看最终test结果前，固定取DASO+validation在质量预算0.01和0.04的prefill/P+D代表点。prefill跑原五个完整likelihood任务，P+D跑原三类完整生成任务；以method-map与canonical权重来源去重。
- 若表示/权重与002已完整评测的策略完全相同，记录来源/哈希并继承原任务结果，避免重复；不能仅凭method-map对新magnitude/Wanda权重误判为旧SparseGPT。
- magnitude/Wanda两个简单剪枝基线固定补充原五个完整likelihood任务，SparseGPT/BF16继承已有完整任务作为锚点。NLL、下游任务分别报告，不将一种预算解释成跨任务精度保证。
- 各完整任务阶段使用空闲GPU并等待该卡前一优先实验结束；不抢占005/006/007/009。全部原任务量、prompt、seed与指标沿用002，不依据test得分挑选任务或更换代表点。

## 006补充：纳入强HAWQ-FP4与phase-uniform mixed对照
- 009结果表明最紧cap的FP4配对解触发最小成本clamp，退化成全NVFP4-prefill/W4A16-decode；其较强性能值得在真实mixed调度中对照。
- 纳入009预定FP4/P+D全部signed/clip/abs和0.01/0.07来源cap，完整map去重后仅两份新策略，不根据mixed结果挑选处理方式。
- 新扩展campaign在同一GPU6重测原六策略与两份新控制的全部三个冻结负载，并在起点/终点各测BF16；以合并10个timed BF16 burst的中位数作锚，保留两个锚点及所有原006测量。
- 原六策略质量输入/方法/权重来源未变，复用已通过的实际B/S/O质量文件；两份新控制按同16个冻结validation窗口、相同forced target完整补测。没有新策略调参，也不改005选择或任何既有NLL/速度记录。

## 006补充复核：测量顺序漂移
- 扩展campaign的standard BF16终点/起点约0.942，接近部分策略间差异，故不能直接将小幅差距解释成稳定优势。
- 在查看额外复核结果之前，固定对全部八策略、全部三负载做反向顺序campaign；仍同GPU6、同输入/权重/seed、2 warmups+5 timed和起点/终点BF16，不挑选有利负载或修改策略。
- 合并两轮每个策略10个timed burst、BF16共20个timed burst，同时保留逐轮比值/锚点漂移；新增27份速度文件，质量沿用已通过且完全相同的24份真实负载文件。
- 该重复用于测量稳健性，不能消除所有热/时钟噪声，也不将5或10次计时区间称为生产硬SLA保证。

## P7 / run011：验证后策略与强化HAWQ的统一标准负载对照
- 005验证后策略与009强化HAWQ分别在GPU2/GPU0计时，不能以跨卡归一化速度直接声称新增策略领先。利用空闲GPU0做统一比较。
- 取009全部35个原定比较点，加入005所有四档质量预算的8个验证选择点；完全相同method-map合并，但保留全部43个比较标签，不根据新计时删点或调参。
- 所有策略已经在原test之前冻结；质量文件在005/009使用相同64个test窗口、B8及相同canonical/scales/backend，按SHA归档并复用，不进行新的质量选择。额外速度seed固定，B8/S2048/O1或64、6GiB KV、2 warmups+5 timed，BF16前后锚点均保留。
- 验收：同卡报告全部真实质量/速度、锚点漂移与实际质量上限切片；明确这是新增验证阶段的效果，不能把新流程冒充原文已有，也不能用测试切片声称形式化最优。
- 首轮011与006反向campaign有部分时间重叠。为排除共享host/PCIe负载混杂，所有其他GPU实验完成后，对同一19map/23组合反向顺序完整再测一次，作为最终主结果；首轮25速度文件及报告/图保留为diagnostic，不混合两轮作为统一环境。策略、质量、seed、warmups/iterations和锚点协议不变，无论结果有利与否不再继续添加重复轮。

## 验收状态
- 已完成本计划内可执行测试及条件分支，004–011全部审计封存，21份人工reviewer证据包副本/链接验证通过，全部GPU作业结束。
- 完整30B预训练模型、生产open-loop与严格真实SLA保证未完成，按本计划预先规定的边界保留；已有shape、搜索规模或均值结果不代替这些结论。
- 结果入口：artifacts/rebuttal/OVERNIGHT_REPORT.md；开发/测量细节追加于dev/impls/144_human_reviewer_rebuttal_overnight_impl.md。
