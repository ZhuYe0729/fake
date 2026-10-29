## 2026-10-27 — 冻结基线保留与部分 decode 精化实验
- 目的：修正旧选点候选覆盖和小样本 gate 的不稳定性；保留原 NLL 约束，不将手工全 W4A16 的已知超限包装为 ε=.01 成功。
- 新增独立 run028，冻结24个 map、原训练代理/LUT SHA、128新 validation、64新 confirmation、两组速度输入。所有新 token 窗口彼此不重叠，并排除旧128个训练/激活校准区间。
- 加入两种 INT8 incumbent、阶段组合和 8/16/32/64/96/128 个 decode 模块的嵌套 W4A16 替换；候选以真实128窗口 NLL gate，使用实测 E2E 作为主目标，另预声明含恢复的 cycle 对照。
- `check_design.py` 检查候选覆盖、只改 decode、正 LUT 收益、token/content 隔离与可行 INT8 保留，通过；所有 Python 文件 AST 语法检查通过。启动3个非GPU0短租约精度 worker；020进入正式速度，028当前主动等待，不抢占父线程。
- 沿用027/016的量化、Graph与计分代码；只复用有效 map 完全相同的权重。自有可重建权重用后清理并留记录；同policy的完整任务串行持锁，避免清理影响兄弟任务加载。
- 文件：plan152、impl152、`artifacts/rebuttal/runs/028/`。确认与正式速度/完整生成尚待执行；不能据此宣称已经改善。

## 2026-10-27 — 完成128窗口选点、独立确认与三轮测速
- 24个固定map全部完成128窗口validation；23个最大预算可行点/预定对照完成单卡calibration，选点先冻结，随后5个配置完成64个新窗口确认。所有新增部分W4A16点未通过ε=.01保护值；两种INT8保留且可行。
- ε=.01的E2E/cycle目标均选均匀SQ-INT8：确认ΔNLL=-.021529，±2SE=.012297，同保护值=-.004233，通过。正式E2E/cycle=1.6317×/1.6316×；普通INT8=1.6329×/1.6328×，两者实质持平。旧NVFP4/SQ点=1.5544×/1.4554×。
- ε=.04两目标均选SQ-INT8 prefill/W4A16 decode：正式E2E=1.8636×、cycle=1.6509×，相对均匀SQ约快14.2%/1.2%。确认均值=.030175未超.04，但mean+2SE+.005=.049408未通过；报告显式区分均值与保护值，不据确认重选，不称为稳健可行成功。
- 原代理自身holdout RMSE=.027399，高于.01预算；三种8/4bit decode的36个o_proj代理贡献全部为0。本轮未修改该代理，也未证明嵌套局部精化优于anchor-only；消融表明确显示两者最终选同一map。
- 完成18个正式速度进程（3轮×BF16首尾+4对照，GPU0/NUMA0，3warmup+5timed），权重切换逐tensor精确相等检查通过。协议沿用016的2120/16384/[8]，不混用019六负载的4168/32768/[1,8]。
- 普通INT8三个完整生成任务在3张空闲卡并行；其只读027权重不由本轮清理，允许分dataset独立锁。自有导出仍按policy串行使用/清理，未产生共享权重清理竞争。新INT8 1000/1500/333分数=22.74715/26.82181/39.58586，Avg=29.71827。相同map的027 Graph全SQ与SQ/W4完整分数带SHA复用，Avg=29.73066/29.92130；不混入002旧eager分数，不宣称下游分数独立新测试或显著性。
- 最终审计：24个validation、5个confirmation、41个calibration/formal测量、3个新完整生成，共73份证据SHA；7个派生CSV及17个Python源码校验通过，所有进程正常退出，无failure记录，自有`model.safetensors`已全部清理，全部代码/元数据/日志/结果保留。015/016原manifest与量化/Graph源码、共享vLLM二进制校验一致。
- 交付：`runs/028/REPORT.md`、`results/selected_results.csv`、`all_validation_candidates.csv`、`candidate_pool_ablation.csv`、`formal_speed.csv`、`full_generation.csv`、`final_audit.json`。本轮是有界选点修正，未覆盖已封存主表，未宣称优于原全部8候选或HAWQ。
