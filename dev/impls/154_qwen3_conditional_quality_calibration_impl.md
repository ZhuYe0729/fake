## 2026-10-27 — 清理后启动固定prefill的条件增量质量诊断
- 用户将DASO本体优化列为优先；按新指示先完成plan153清理45份可重建权重/373.286 GiB，保护118423份文件后才启动本轮测试。
- 新增可选核心模块`daso/modeling/conditional_quality.py`，由相同窗口的BF16、SQ基线、单组/双组整模型NLL校准signed增量与交互；保持原`quality.py`及其源码SHA/default行为不变。
- 三个有意义的CPU测试验证BF16锚点还原、signed改善、未拟合组合、pair交互和窗口形状，红阶段缺模块复现、实现后全部通过；原modeling三个测试通过。环境未装pytest，因此使用可由pytest同样识别的unittest，不新增依赖。
- 冻结run029的16组/8pair、36个校准map、26拟合policy与8未拟合policy。校准使用旧训练128块的前64块并跑真实Graph；272个新窗口用于128validation+128confirmation+16速度输入，排除原128训练及028全部208窗口，token区间和内容隔离检查通过。
- 四种代理比较：旧代理仅anchor修正、原架构相同新26标签重拟合、条件单组、条件单组+pair；只用固定8个policy holdout选主版本，再枚举65536组mask生成至多20个proposal/anchor。最终128窗口真实gate与独立确认均未读取，未放宽预算或保护规则。
- 使用3张空闲非GPU0卡开始36个校准任务；所有阶段持现有资源锁，CPU拟合隔离到NUMA1且禁CUDA；候选权重只在需要的阶段暂存，最终清理并留记录。当前仍在校准，尚不能宣称代理或部署收益改善。
- 文件：plan154、impl154、可选核心模块/测试、`artifacts/rebuttal/runs/029/`；原封存结果与共享vLLM/kernel不改。

## 2026-10-27 — 029完成并独立审计
- 36个64窗口Graph校准map完成；固定8个policy holdout选择条件加和版本。MAE：旧anchor修正.014972、原架构相同26新标签重拟合.013488、条件加和.002201、pair.003361。未因pair更复杂而强行采用。
- 新128-window validation共同5个SQ子空间map的MAE：旧.027865、原架构重拟合.025141、条件加和.010301、pair.016876；代理改善跨新窗口仍存在，但剩余误差不可忽略。
- 枚举65536mask后仅7个唯一validation候选，6个校准测速点。ε=.01仍SQ-INT8，ε=.04选择`group_7fff`（135模块W4 decode，最后9层down_proj保留SQ）。主版本选择先冻结，再打开128个新confirmation。
- `group_7fff`确认ΔNLL=.000841±.013767（2SE），保护值=.019608≤.04；两预算均值/保护值均通过。三轮GPU0 E2E=1.7961×，相对同轮INT8快10.33%；cycle相对INT8=.9873，含恢复反而约慢1.3%，不宣称全部指标最优。
- 135-W4完整CNN/DM/DialogSum/IWSLT=22.77579/27.16597/39.69811，Avg29.87996；当前Graph INT8=22.74715/26.82181/39.58586，三个分项均更高但未作显著性宣称。SQ与full-W4/INT8对照带SHA复用027/028，相同协议，不混旧eager。
- 75份测量/评分证据、源码与CSV哈希复核通过；原结果/核心文件SHA一致，所有进程正常结束，无failure记录，自有checkpoint二进制已清理。新增可选核心源码独立归档到reference/core及core_snapshot.json，方便后续修改时重放旧版本。
- 交付`runs/029/REPORT.md`及逐组、holdout、新窗口误差、确认、速度、完整生成CSV。因均值提案仍没有覆盖ε=.01保护值可行的快速点，后续155专门修正提案约束，029结果保持封存。
