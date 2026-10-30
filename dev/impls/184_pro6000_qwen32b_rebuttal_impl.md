# 184 — PRO 6000 完整 Qwen3-32B rebuttal 实验

## 2026-10-29 - 交接核对与独立实验准备
- 已读根AGENTS、PRO6000_HANDOFF、中文回复总稿及后端/复现契约；确认当前缺口是完整32B质量/E2E，旧053已结束。
- 沙箱内NVML/CUDA失败，经授权任务范围内的沙箱外只读检查确认两张RTX PRO 6000 Blackwell Server Edition（97887MiB、SM120），当时各2250MiB驱动占用、利用率0%、无计算PID。
- 本地找到Qwen3-32B的17个权重分片；diffusion环境为torch2.13+cu130/transformers4.57.6，cospaq为torch2.9+cu130/transformers5.9，均无vLLM，需独立准备，不能直接宣称重现归档环境。
- 本轮计划184与新run055独立保留，不改旧结果与共享环境。

## 2026-10-29 - 完整模型可运行与锁定环境恢复
- 17个BF16分片逐文件SHA256、连续offset、707张量与448个投影shape检查通过；共32762123264参数、65524246528权重字节。完整64层在GPU0完成16步KV生成，全部参数BF16/CUDA；HF可运行性探针峰值65572514304字节，不作为正式phase速度/质量表。
- 用户提供`/root/wja2/project/my/vllm-cospaq`；核对其HEAD为679527f，复制到run055独立依赖目录后应用归档补丁，三处修改逐文件SHA与归档快照一致，不修改用户的原clone。
- 独立.venv安装torch2.9.0+cu128、transformers4.55.2、vLLM0.11.1；官方wheel的7个预编译扩展链接到本轮fork副本，记录二进制SHA。归档Transformers低于官方wheel元数据要求，使用显式override且LLM/phase模块导入通过；后续仍需实际GPU验证。
- 下载公开WikiText并用32B tokenizer重新编码，train/validation/test三份完整token tensor与001逐token相同；校准128窗seed0、validation100窗seed1、test100窗seed42，各2112 token。样本与选择/正式测试分开；保存新采样起点和SHA。
- 单独安装CUDA12.8.93工具链；CUTLASS锁定e406c18的必要头文件稀疏检出成功。源站超时/中断的所有安装与下载日志保留；未采用失败的完整源码归档。
- GPU1已启动32B自己的128窗SparseGPT 2:4 canonical准备。新增055显式路径入口、初始B8/S2048/O1或64设计和256个fused模块策略；正式测量前仍需完成后端/容量试运行及策略冻结。

## 2026-10-29 - 32B原生vLLM容量通过，正式数值准备推进
- 完整32B在锁定fork上通过B8/S2048/O64、BF16 KV、FULL_DECODE_ONLY Graph、固定8GiB KV；峰值分配77506812928字节、峰值预留78492205056字节，单卡可容纳正式目标负载。
- 完整BF16已通过同一phase exporter导出256模块；当前进行100窗validation Prompt/Decode NLL，质量准备不作为正式速度。首次验证启动曾与原生容量探针接近，已在模型加载前终止该次phase进程，并改用GPU0文件锁串行化；该次日志保留。
- 32B自己的128窗2:4 SparseGPT canonical状态完成，64层剪枝约819秒；随后GPU1串行启动配对4:8状态。GPU0排队采集新32B激活尺度和plain INT8/SQ INT8/FP8局部误差。
- Dense NVFP4、Sparse NVFP4、Marlin NVFP4的6项数值用例已通过；Sparse BF16唯一失败为wheel仅有带版本号库导致链接找不到-lcublas/-lcusparse，已在独立CUDA前缀补开发链接并只重跑该用例。
- 新增055 README。临时策略权重按计划使用本轮专属内存盘以避免重复写数十GiB；不删除原始模型、canonical、代码、日志或测量结果。

## 2026-10-29 - 按用户要求集中三种方法
- 用户将正式完整评测收窄为BF16、DASO与HAWQ-V3，其他baseline先不测；计划184同步更新。候选局部误差和uniform策略仅保留必要训练/校准，不额外跑其正式test或下游任务。
- 统一phase BF16 validation已完整完成：100窗Prefill平均NLL 2.1151909155611346；100窗真实KV continuation平均NLL 1.9559595648871484，均非正式test结果。四后端共7项数值用例全部通过。
- 两套新32B canonical状态均完成；8bit激活尺度与局部误差64层采集约211秒。32B的448个真实NLL Hessian对角块迹已测完（P4/B2/S128，每模块8观察），约451秒、GPU峰值约73.47GiB；明确不冒称原8B的P16/B4/S256协议。
- 固定72套32B校准策略（54 train/18 holdout）及两个代表性质量预算0.02/0.07；只用train策略拟合，validation作闭环选择，正式test尚未开始。

## 2026-10-29 - 完成成本画像并启动完整校准
- 全部3584条局部误差、32B对应64项成本画像（8后端×4矩阵×2阶段）通过完整性检查；画像在两卡无计算PID、显存均2250MiB驱动占用时串行执行，GPU0固定NUMA0/CPU0–7，约20.7秒。
- HAWQ主迹改为旧Qwen协议P16/B4/S256，在任何HAWQ新质量标签之前冻结；P4/B2/S128只作试运行。主迹GPU0已启动，峰值约86.04GiB；GPU1运行固定72策略真实KV校准。
- 首次校准试运行发现128窗train取100窗时尾批多取4窗，未接受任何104窗标签。已在加载后严格截取100窗，验证三split尾批均4窗；新试运行已获得BF16/FP4/稀疏BF16完整100窗双场景标签。补充逐批partial证据，后续失败不会丢失已记录窗口。
- SHA校验曾与模型加载竞争磁盘，临时挂起本轮自有校验进程，待模型进入GPU计算后以idle IO优先级恢复；两套canonical SHA均已完成，未改任何权重。
- 下游准备：按原61份task配置冻结公开数据，MMLU test共14042题；原生成1000/1500/333条输入验证SHA/唯一ID/长度/vocab范围。补lm-eval0.4.12、ROUGE0.1.2、SacreBLEU2.6.0，constraints确保未升级已有数值包。
- 环境差异显式写入environment_comparison.json：实际C++为官方vLLM0.11.1 wheel，与旧dev版本不同；datasets/scipy/pandas等辅助版本差异单列，不宣称与旧归档二进制相同。32B所有质量系数和延迟重新测，三方法在同一新运行时比较。

## 2026-10-29 - 完整HAWQ主迹完成与双卡校准交接
- P16/B4/S256主迹完成全部448个原始投影块，每块64观察，4338.3秒；保留76个负迹块及signed目标，不按对照胜负切换迹。
- 校准已完成17/72套。旧GPU1调度父进程在cal32_p16子任务正常退出且100窗双场景/权重审计闭合后停止；交接JSON保留返回码与SHA。新增逐策略非阻塞锁，两卡从同一固定manifest领取未完成策略，不重复收集。
- NVML v2实测空闲allocated为2359033856字节/卡，reserved为668139520字节/卡；与最初2250MiB空卡观测匹配。单进程忙时未归属gap为2255.0625MiB。修复监控采集器MiB文本解析及reserved重复计数，保留失败日志；正式监控使用独立实测offset而非将驱动常量当外部干扰。
- GPU1共享队列已恢复，GPU0先核验cal32_p08同Prefill/不同Decode的100窗标签复用，再自动加入共享队列。均固定对应NUMA和8核；临时导出锁串行、成功结果保留身份后仅清理新建权重二进制。
- 在正式工作点选择/速度之前补充冻结三轮聚合规则：轮内10次中位数，本轮BF16首尾中位数均值归一化，三轮比值均值，>5%波动标记且不删轮。
- 新增continue_run.py接续控制器，等待72套与复用检查后顺序执行拟合、validation闭环、HAWQ匹配、冻结、test、独占三轮测速及双卡完整下游；失败停止并留日志。新增final_report.py从原始逐窗/逐轮/任务闭合记录独立核验并生成中文表和CSV；当前尚未运行完整报告，不能据此声称正式实验已完成。
- Prefill标签复用实测通过：cal32_p08原phase策略（GPU1）与相同Prefill、Decode绑定为Prefill的新策略（GPU0），100窗最大绝对NLL差为0；GPU0共享worker已开始执行。交接入口同步指向055实时状态，补存安装任务包后的独立pip freeze，旧freeze保留。
- 终审报告将重算三轮计时中位数/归一化比值，核验每策略100窗和尾批4、真实Decode逐token capture、任务样本数与生成ID/分片覆盖，保留Torch与NVML不同内存口径及每个策略模块组成；未测结果不会生成完成报告。

## 2026-10-29 - 交付时区与目标确认
- 用户希望尽量北京时间明天（2026-10-30）拿到完整结果，未指定硬性时刻；配置delivery_target.json与计划同步记录。
- 双卡校准/接续控制器持续运行，正式三方法质量—速度—内存闭合后接完整下游；评分/终审成功才自动写入中文总稿，阶段失败会保留日志并停止接续。

## 2026-10-29 - 夜间资源利用与验证集多点搜索
- 用户追加十几个小时夜间窗口，并希望多找几个DASO优势点。新增独立search_extension.json，原72套校准manifest及其策略SHA不改；在DASO/HAWQ validation/test前扩展六档搜索预算、四组阶段组合，正式仍保留两档预算。
- 新增quality_pool.py，将DASO/HAWQ validation和正式test NLL分配给两卡；每GPU锁+逐任务锁防重复。数值experiment.py保持不变。接续控制器改为由各阶段管理自己的GPU锁，避免外层GPU0锁与worker死锁。
- 新增scout_latency.py：质量通过后固定shortlist，再独占GPU0做一次探索E2E/cycle，独立于后续冻结后三轮。实际E2E而非仅Linear代理用于最终选点；所有候选质量、探索计时及不利点保留，完整下游仅对正式三方法代表点运行。
- CPU合成集成核验通过：两个worker领取7项无重复；shortlist按映射去重、排除质量不合格项；构造代理成本/真实计时排序相反的用例，选择器正确选择实际E2E更低项。记录为synthetic预检，不能算作32B实验结果。
- 仅在旧接续控制器仍等待校准、确认无数值子进程时替换它；交接记录continuation_parallel_handoff_1793275463733479035.json。新控制器PID3206475、日志continuation_driver_parallel_closure.log；原两张GPU校准worker持续运行，未停止其数值子任务。新控制器质量阶段不持外层GPU锁，交由两个worker各自持锁；探索与正式测速同时持两卡锁。
- 为便于在较长下游任务结束前使用结果，新增core_summary.py：仅当100窗test和正式三轮速度已闭合才生成CORE_RESULTS.md/core_results.csv，复用预先核验的配对NLL与三轮归一化规则；完整原任务Avg和终审仍等待下游结束，不提前写入总稿。

## 2026-10-29 - 训练标签完成与CPU拟合预检
- 54套train策略标签已闭合，18套held-out策略继续由两卡采集。利用空闲CPU8–15、CUDA_VISIBLE_DEVICES为空，仅对54套训练标签做原QualityCalibrator预检，不读取held-out/validation/test。
- Prefill训练MAE0.010067、bias−0.002470；P+D训练MAE0.009043、bias0.003087 nats/token；未见阻止两档预算搜索的拟合异常。结果位于results/fit_training_diagnostic，独立保存，不能解释为留出泛化指标或正式完整模型对照。
- 正式72套流程与拟合逻辑未改，待留出采集完成后再产出完整拟合/留出误差和六档代理解。
- 在实际DASO/HAWQ validation开始前，将探索计时简化为覆盖全部质量合格的不同候选映射，避免代理top3漏掉切换成本更优的点；重复映射仍去重。规则旧版快照保留，CPU用例验证可覆盖代理排名之外的质量可行点且排除不合格/重复点。

## 2026-10-29 - 全部校准完成、留出误差与嵌套绑核修复
- 72套校准全部闭合，72个不同完整映射、54 train/18 holdout无映射重复。正式拟合的held-out策略MAE为Prefill0.012779、P+D0.008832 nats/token；仅为同固定train文本上的留出策略误差，不能作test/任务精度替代。
- 六档预算两场景共12项代理解完成，验证池为20个策略job/24个场景记录（含既有BF16 validation复用）；已开始双卡真实100窗validation。
- 首次质量pool启动在Python之前被numactl拒绝：父控制器CPU8–15绑定使默认libnuma范围过滤掉子任务的CPU0–7/32–39。为嵌套numactl加--all后重新显式设置CPU/NUMA，实测子进程分别为0–7/bind0与32–39/bind1；同步修复探索/正式测速与下游的同类子进程启动。失败日志保留，无新质量标签被接受或覆盖。
- 新控制器PID3220616、日志continuation_driver_affinity_retry.log；fit和prepare_validation的已完成记录复用，不重新选模型或改变数值后端。两卡验证均已实际计算。
- 终审增加校准策略文件SHA及72映射去重检查；准备成本补报首次校准命令到末次闭合标记的观察墙钟（含失败/导出/加载/编译/等待），不冒充GPU计算时间。中文总稿更新前留原稿快照，并在最终完成后更新055/交接入口状态。
- 报告显式处理HAWQ代理延迟不可行的预设分支：若发生放宽，在主表/核心表/总稿标‡，CSV保留请求与实际上限，不将其表述为相同代理上限内的对照。该标注仅依据实际allocation标志生成，不删或调整测量点。

## 2026-10-29 - 真实validation反馈与四个紧预算回退点
- 原20策略/24场景validation完成。原DASO优化候选在报告预算0.02下未通过质量余量；uniform FP8可行，但希望找到更快的混合点。用户已授权效果不好增加搜索点。
- 冻结原质量模型和actions不变，用相同两步MILP追加0.003/0.005两档，两场景4点。预测Linear成本：Prefill1382.2/1359.2ms，P+D2655.4/2446.8ms；这些仅代理预测，尚非正式实测速度。
- 新增extend_low_budget_validation.py，将4点分双卡真实100窗验证后合并候选池；新规则明确记录这是validation驱动的预算回退，原六档配置/规则/探索shortlist快照保留，不伪称全部候选在初始validation前预注册。正式test/下游均未启动。
- 暂停的仅是自有调度父进程；初始探索中的活动数值子进程正常完成、退出码0、15样本与精确权重审计通过，原始基线/该点测量和补充资源记录全部保留。成功自有临时权重按已保存SHA清理，随后退出CPU父进程以便扩展完整探索。控制记录superseded_scout_child_finished.json。
- 低预算扩展驱动PID3225639、日志low_budget_validation_driver.log；该脚本完成后将自动接续continuation_after_low_budget.log，原拟合/prepare/validation阶段已完成记录不重跑。
- 原0.02/0.07报告档保留，增加0.04中间档；在正式test/下游开始前记录reporting_budget_extension.json。探索测量集合按最大预算0.07筛选，新增中间档不改变任何待测映射或计时程序；原配置/shortlist快照保存，points SHA前后一致。报告循环和满足率改为读取实际预算数，相同映射共享证据，不膨胀独立样本数。

## 2026-10-30（北京时间）- 完成探索与正式选点
- 额外4个低预算点的100窗validation全部闭合；完整探索共20次模型运行，起末BF16 Prefill为3019.94/3020.09ms，P+D为6259.02/6259.65ms，未据胜负筛轮。
- 六个DASO预算/场景点已按质量guard及独立探索E2E确定：Prefill选q0.003、q0.02（后者共享0.04/0.07）；P+D选uniform FP8、q0.01、q0.07。uniform选点不解释为层异构收益，共享映射不当作独立重复。
- HAWQ主迹适配求解完成并开始双卡100窗validation；0.04/0.07对应代理上限在HAWQ候选集不可行，按预设放宽到其最小可行延迟，报告保留‡及请求/实际上限。所有正式test、三轮速度、完整任务仍待完成。
- 当前接续控制器PID3226470，日志continuation_after_low_budget.log；没有重启旧053或改变数值后端。

## 2026-10-30（北京时间）- 正式NLL与三轮速度闭合
- 全部11个不同策略/场景的正式test各100窗闭合；DASO六个预算设定的平均ΔNLL均在对应预算内，相同映射共享证据。
- GPU0独占三轮正式测速完成，共33次模型运行；两卡锁、固定CPU0–7/NUMA0、90秒空闲准入、每case5暖机+10计时、BF16首尾对照和第二轮反序均保留。未出现资源干扰重跑记录。
- CORE_RESULTS.md/core_results.csv已生成，最高DASO P+D E2E/cycle为BF16的2.189×/1.902×；Prefill较宽两档为1.771×，Torch分配峰值与Host缓存分列。HAWQ放宽代理上限的‡标志保留，不宣称所有点同时占优。
- 正式策略冻结SHA 9a06a2b4458f2141b8a3bd7db0633959afe61e268e5f66a14e79c6f68047488d；下游双worker已启动，共61任务/分片，完整Avg与最终独立审计尚待完成。

## 2026-10-29 - 三方法完整对照闭合
- BF16、DASO三预算及HAWQ匹配点完成冻结后的100窗test、三轮独占速度/内存和完整原协议下游；相同策略共享证据，不计为独立重复。
- final_report独立核验结果SHA、逐窗/逐token记录、计时中位数、任务计数与分片覆盖，生成R4中文结果、CSV和来源清单；同步中文回复R4规模段与内部核对，完整保留负面点及波动。

## 2026-10-30（北京时间）- 交付索引与无损归档
- 全部61个完整任务/分片与最终来源审计已通过，中文总稿R4第2点回填完整32B表；响应索引、精选证据、实验清单和交接中的“完整32B未完成”已更新，持续错峰服务/双worker PD仍保留未完成范围。
- 三生成均值：BF16 29.730，DASO中档29.865/较快档29.398，HAWQ较宽点29.626；Prefill四项均值：BF16 74.745，DASO紧档74.298/较宽档73.448，HAWQ紧档73.852/较宽档73.811。不得概括为所有点同时占优，含恢复排序仍分列。
- 使用原归档工具make_archive新增run-055组，10,122个原始文件共3,029,194,957字节，18个≤40MiB分片共749,906,655字节，原文件未删除。restore_raw_results.py --group run-055完整恢复/SHA核验通过；旧组及其分片不变。
- 服务重启后核验原数值进程继续运行，没有重复启动；最终刷新status.json，全部完成标志与continuation complete一致。未提交或推送Git。

## 2026-10-30 - Git交接整理
- 用户要求提交文档、开发记录、实验代码、配置、日志和结果，供原机器pull后整理rebuttal；目标为origin/main。
- 核验055原始证据18个分片SHA及10,122文件清单，最大单文件40MiB；保留小型token/采样tensor，不纳入模型权重、canonical权重、导出模型、.venv、依赖副本及编译/数据下载缓存。
- 提交前核对远端基点与本地HEAD一致；新增文本未发现凭据模式。沿用现有run-055恢复入口，Git提交不改变数值实验、冻结策略或报告结果。
- Git范围复核补纳入1.69MB validation token tensor与5.72MB activation scales统计；二者用于恢复冻结输入/量化尺度，均非模型权重。约975MB画像权重及两套约65.5GB canonical权重保持排除。
- 原始CSV的CRLF、上游兼容头及日志空白保持原样以保全SHA；对本轮Markdown/Python/shell/CSV（识别CRLF）执行暂存区空白检查，不改写冻结源码快照。
