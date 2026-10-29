## 2026-10-26 - 启动一致运行时重分析
- 目的：在所有方法共享正确、可比较的执行路径后更新速度分析；不把 uniform Graph 与异构 eager 混排。
- 新建计划148和 run015；原始质量与速度证据保留。
- 实现路线：稳定 GPU 地址的按层复用缓冲区，保留异步阶段搬运；先数值/地址/显存验证，再画像、选点和统一测速。

## 2026-10-26 - 运行时与画像验证通过，进入重新选点
- `arena_runtime.py` 用按层 max(prefill,decode) 字节缓冲区固定地址、异步事件顺序复用；非双驻留。原 eager / arena eager / arena Graph 三个真实异构 map 的完整生成 tokens 一致，prefill/decode 搬运权重逐张量精确相同。
- 额外发现 dense/sparse FP4 GEMM 的 alpha.item() 强制 GPU→CPU 同步，无法 Graph。隔离复制绑定源至 run015/code/csrc，改为 CUTLASS 读取同一 FP32 device alpha；初始化/运行使用当前 stream，Sparse workspace 使用 torch 分配，原量化、布局、tile/累加类型不改。24个形状×动态alpha对照均0误差；修改后3个真实异构 map 再次逐张量/生成一致。
- 新画像80项完成：10种方法 ×4种Linear × M8/16384。Decode以512MiB左右的多份独立权重工作集捕获，避免单个小权重重复命中L2；仍是加和代理，不当作整模型E2E。
- 新分配池最初70标签/50 map；在读取新validation/test分数前，按训练代理和硬件画像的非支配decoder集合（BF16/W4A16/SQINT8）补齐7×3阶段统一anchors，新增14 map。初始池/规则/任务清单保留，修订原因和数据使用边界写入JSON；未删除任何候选。
- 质量复用按有效map、同一canonical权重、冻结64窗口数据SHA及batch8核验，已有13个validation场景继承；新增49个场景实测。GPU1/2/5因其他任务占用，3次启动按显存门槛退出，已保留日志并恢复到0/3/4/7。`finish_validation.py` 衔接补充anchors与安静GPU0校准、冻结选择。
- 实际发现Sparse BF16源码cache上限固定512，既有环境变量并未被读取；新记录区分requested环境值和实际源码，不宣称已把cache限制为8。
- 保留3类调试失败：arena恢复时原CPU布局与finalize后flatten布局不一致（reshape后通过）；初始画像释放未定义out（修复）；FP4旧绑定捕获不支持（device alpha修正）。共享vLLM和原始kernel源码未修改，局部编译在 /tmp，旧001–014保持。
- 后续：完成校准/冻结选择后，只对改变的main map补完整质量任务；统一重测新旧固定配置和实际约束，更新主表及开销。

## 2026-10-26 - 校准冻结与可行HAWQ对照
- 62项validation与55个候选map校准完成；DASO选择文件已冻结，不使用新test选点。
- 新画像下，原配对HAWQ上限均不可行并产生重复最小成本点。保留旧设计至run015/reference/pre_feasible_hawq_controls，新增12个HAWQ延迟控制点：BF16校准E2E的.95/.80/.65，经BF16加和偏移与5%余量换算线性上限，全部可行。主表预定使用.80，不按test挑选。DASO候选池/选择保持不变。
- 更新freeze_main/prepare_full_quality；主表17行需新增11组完整质量评测，共39个policy质量job、45项新test NLL、42个正式速度policy（三轮129进程）。
- 首轮质量worker1/2/3因嵌套numactl继承CPU0–7而拒绝其他CPU，保留失败日志；launch增加--all，保留有效worker0，恢复其余worker到空闲GPU1/4/7。

## 2026-10-26 - 下游完整评测与补充负载流水线
- 数值/地址gate通过后，新增main完整质量与test正在GPU0/1/4/7运行；test请求清单已冻结。新增HAWQ可行控制仅用于baseline，不回写DASO选择池。
- 冻结三个固定map迁移负载：B1/S2048/O128、B8/S512/O128、B8/S4096/O64；11个强基线及DASO版本，两个正反顺序campaign、BF16首尾锚点。明确不把固定map迁移当作每负载重优化或质量保证。
- benchmark增加batch/prompt显式参数，主表默认B8/S2048及2120模型长度/16384批token配置保持；新增workload_queue/analyze_workloads，接在三轮主表之后安静运行。
- 新增全部HAWQ预算/不可行点分析、准备与运行时开销分账、最终manifest审计。原始失败/重复点均保留。

## 2026-10-26 - HAWQ Hessian估计的符号敏感性
- 原始252个逐层trace估计中35个为负。为避免只用signed约定的弱基线，在不查看新HAWQ test分数、不移动主表行或DASO选择的前提下，沿用run009的clip_zero/absolute两种处理，固定中心.80预算补充8标签（6个policy job）。
- 所有补充点均报告独立test NLL和三轮速度，不按test选符号处理或删点；主表仍为signed约定。原design备份于reference/pre_hessian_robustness_design.json，正式速度现在48个policy、三轮147进程。
- 新增hawq_robustness_queue，接在完整主表质量作业后、正式速度前；同一map的重复标签只评测一次，保留全部标签来源。

## 2026-10-26 - 完整来源与运行环境检查
- 001–014共56,435个manifest条目逐项核验通过。新增完整质量已有27个指标cell用原run013解析器核对样本数/数据SHA通过。10个正向精确MILP均status=optimal，最大数值gap约1.5e-15。
- B1异构Graph/eager四次生成tokens相同、阶段搬运全部权重精确一致；B8/S4096 BF16预检查可行。smoke仅作数值/显存门槛，不参与速度排序。
- telemetry按CUDA_VISIBLE_DEVICES读取实际物理GPU，正式速度追加计时外其他GPU利用率/显存采样，不影响计时区间，也不筛选他人作业期间的样本。
- 补充说明test沿用旧冻结窗口，validation/test为不同split，但历史已公布标签不能宣称从未查看过。单次新run选择冻结仍在新test评估之前。

## 2026-10-26 - 新主表完整质量完成
- 新增11组完整任务已完成，38个指标cell通过原run013解析器的完整样本数、指标名和输入SHA检查；记录full_quality_metric_preflight.json。主表其余相同有效map质量按已有来源继承。
- 符号敏感性补充利用空闲GPU2/3/4/5提前完成全部test NLL，未与正式速度并行。39个main/raw test job仍在收尾，完成后自动进入GPU0三轮正式速度；速度未完成前不发布新排名。
- 下游分析、三个代表负载、开销、五位人工reviewer结果包与最终审计均已接入post_selection自动流水线；保留原失败父进程状态，通过recover_full_quality衔接。

## 2026-10-26 - 优先发布主表
- 用户关心等待时间。旧run013正式主表69进程，成功测速进程合计约54min、模型导出约30min；本轮原先48个policy/147进程把25个补充policy与主表混排，导致主表需等待全部对照。
- 不改map/预算/选择/质量/每轮3warmup+8timed或三轮重复。将23个主表policy的三轮72进程排在前面，完成即用build_report --main-only发布17行187指标；25个补充policy随后三轮，分别增加自己的BF16首尾锚点。新设计归档旧版本reference/pre_main_priority_design.json。
- 原round0已完成的主表数据保留使用；原先交错的补充前缀保留作诊断，补充组使用_extra独立record-ID重新测，避免跨数小时沿用主表BF16锚点。额外组新增锚点后81进程；分析通过report_paths显式选择对应record和anchor。
- 调度在当前单配置完成后切换，不中断正在运行的测速，不删除已有结果，也不按输赢选取数据。
- 补充计时口径：run013成功测速命令起止跨度约107.9min；前述84min为测速/导出进程累计耗时，不等于墙钟跨度。改序后已有主表记录通过原解析器的输入SHA/样本数/首token计时检查，主表/补充record与锚点映射、两种发布入口语法检查通过。

## 2026-10-26 - 主表已发布，补充组继续
- 主表72/72个正式进程完成，17行187个指标已更新qwen3_table2/README.md、CSV、LaTeX及逐项来源。复核17行数值完整、188个来源文件SHA一致；补记priority_main_published.json。
- W4A16 NVFP4/Marlin的TPOT约1.83/1.82×；DASO epsilon=.04 prefill约2.21×，HAWQ最小成本约2.19×。P+D五候选DASO约1.90×、HAWQ最小成本约1.87×，重复范围重叠且前者任务质量较低，不据此宣称全面领先。
- 刷新质量检查至所有10个预算点完整，原中间文件保留reference/quality_checkpoint_before_completion.json。补充组及负载迁移仍继续，最终违约/开销/封存报告待全部完成。

## 2026-10-26 - HAWQ比较边界审计
- 用户询问是否可合理减弱baseline。核对官方论文与本地分配代码，新增run015/HAWQ_COMPARISON_AUDIT.md；不改主表数值、候选/预算或正在运行的任务。
- 记录plain INT8与SQ INT8动作集合不一致、FP4阶段专用后端适配、最小成本端点原加和约束不可行且Hessian目标几乎无选择空间。不能由代理约束不可行推断实际E2E不可行，也不据此删除强结果。
- 后续建议为同候选同约束分配器对照、同算子阶段消融和DASO预测/覆盖诊断；尚未新增这些实验，不预设削弱或改进一定带来胜出。
