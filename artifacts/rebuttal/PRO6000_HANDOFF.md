# PRO 6000 大模型实验交接

更新：2026-10-30（北京时间）。用途：在另一台机器 pull 后，让 Codex 直接了解现有结论、代码入口与待补实验。初始交接不包含模型权重、编译缓存或可重新下载的数据集缓存；本机随后启动的新实验见下方055入口。

## 当前 PRO 6000 工作入口（055，已完成）

- 用户已将正式完整对照收窄为 **BF16、DASO、HAWQ-V3适配**，其他baseline暂不独立评测；uniform仅作必要校准/选点锚点。
- 用户提供的fork位于`/root/wja2/project/my/vllm-cospaq`，HEAD核对为679527f。本轮使用055内独立副本及环境，原clone未修改。
- [055 README](runs/055/README.md)、[计划184](../../dev/plans/184_pro6000_qwen32b_rebuttal_plan.md)、[实施记录184](../../dev/impls/184_pro6000_qwen32b_rebuttal_impl.md)是新机器入口。完整Qwen3-32B的BF16容量/validation、两套canonical、kernel验证、局部特征、PRO6000画像、HAWQ主迹、72策略校准及DASO多点validation/探索测速已完成；正式100窗test、三轮测速、61个完整下游任务/分片与独立终审全部完成，见[完整表及原始证据](responses/reviewer4/pro6000_qwen32b_run055/README.md)。
- 用户目标为尽量北京时间2026-10-30拿到结果，夜间有十几个小时；要求更多DASO候选点。055在任何DASO/HAWQ validation/test前追加六档预算、四组阶段组合与独立探索E2E选点；其后根据紧预算validation反馈，用原模型追加0.003/0.005搜索，正式报告保留0.02/0.07并增加0.04中间预算；这些扩展均在正式test前。质量闭环/test双卡，正式三轮仍独占。
- 先读[实时接续阶段](runs/055/results/continuation_status.json)和[保存结果状态快照](runs/055/results/status.json)，再检查实际自有进程/日志；不要重复启动正在运行的校准/接续控制器。正式工作点、test/速度及任务未闭合前，不向中文回复填写未测数据。
- 入口`source artifacts/rebuttal/runs/055/code/env.sh`；当前沙箱内CUDA不可见，GPU执行须走主机通道。两张卡可并行质量准备；正式测速须等待两卡准备结束并同时持锁，固定GPU0/CPU0–7/NUMA0。

## 先读这几份文件

1. [中文 rebuttal 总稿](REBUTTAL_RESPONSE_CN.md)：五位人工 reviewer 的问题、现有答复及新增完整32B结果表。
2. [当前精选证据](REBUTTAL_SELECTED_EVIDENCE.md)：结果入口及测量口径。
3. [原审稿意见](reference/reviewers.md)与 reference 中的论文；不处理 AI reviewer。
4. [实验清单](EXPERIMENT_CHECKLIST.md)和[实验索引](responses/README.md)。
5. [开发记录181](../../dev/impls/181_qwen_nine_hour_idle_watch_impl.md)、[182](../../dev/impls/182_chinese_rebuttal_consolidation_impl.md)：最近实验结束状态及回复口径。

## 目标与优先级

055已补齐完整Qwen3-32B在PRO6000上的三方法对照，支持R4的30B级规模问题。下述准备流程保留供复现；正式范围及完成状态以上方055入口为准，新机器仍须重新核对型号、资源和数值环境。

1. 先检查 GPU、CPU/NUMA、CUDA、PyTorch、vLLM 和 kernel；确认 BF16 完整模型能在目标负载下运行。
2. 优先完成 BF16、可靠8bit基线、HAWQ-V3适配及 DASO 代表点的真实质量/速度对照。DASO 与基线使用同一硬件、数值后端、任务协议和负载；小规模试运行通过后再确定正式 batch。
3. 新模型须重新采集敏感度/质量校准；新 GPU 须重新画像延迟。不能把8B质量系数、已有8B策略或5090速度直接改标签当作32B结果。
4. 先做可比较的 NLL、TTFT/TPOT/E2E、含恢复 cycle 和内存，再按时间补原协议下游任务及长 Decode。工作点在正式测试前冻结，完整保留全部尝试和异常。
5. 旧机器封存至054，本轮055已独立完成。后续新实验使用未占用编号，按根AGENTS.md写计划与开发记录，不覆盖旧运行。

当前112个 Qwen3-32B shape forward 仅用合成权重，见[004报告](runs/004/REPORT.md)，不属于完整32B质量/E2E结果。这一完整模型缺口已由055补齐。

## 拉取与依赖

在已有克隆中执行正常 pull，保持本地修改安全；不要直接重置已有工作区：

~~~bash
git pull --ff-only origin main
git submodule update --init --recursive fake/kernels/cutlass/cutlass_wrapper third_party/sparsegpt third_party/smoothquant third_party/lm-evaluation-harness
~~~

本次补上了 CUTLASS wrapper 在 .gitmodules 中缺失的映射。wrapper 固定在 8f60a9d2251e39fc7adbb385f6612260db56ec38，按照其 INSTALL.md 在目标机器重新编译；不会搬运当前机器的 .so、.o 或 Triton 缓存。

**实验真正使用的 phase vLLM 不是 third_party/vllm。** 它来自独立 fork：

~~~text
https://github.com/ZhuYe0729/vllm-cospaq.git
679527f6fffdc51ee79a92b4d9b0749a8855e5b6
~~~

原机器实际路径为 /root/wja/project/my/cospaq/test/vllm。当前相对该基线的三处修改已保存为[工作树补丁](portability/dependencies/phase_vllm/working_tree.patch)，关键源码快照也保存在[files目录](portability/dependencies/phase_vllm/files)。可以另行克隆，然后在干净基线上应用：

~~~bash
git clone https://github.com/ZhuYe0729/vllm-cospaq.git ../vllm-rebuttal
git -C ../vllm-rebuttal checkout 679527f6fffdc51ee79a92b4d9b0749a8855e5b6
~~~

随后令 REBUTTAL_REPO 为本仓库的绝对路径：

~~~bash
export REBUTTAL_REPO=/absolute/path/to/fake
git -C ../vllm-rebuttal apply --check "$REBUTTAL_REPO/artifacts/rebuttal/portability/dependencies/phase_vllm/working_tree.patch"
git -C ../vllm-rebuttal apply "$REBUTTAL_REPO/artifacts/rebuttal/portability/dependencies/phase_vllm/working_tree.patch"
export DASO_VLLM_ROOT="$(realpath ../vllm-rebuttal)"
export DASO_CUTLASS_WRAPPER_ROOT="$REBUTTAL_REPO/fake/kernels/cutlass/cutlass_wrapper"
~~~

然后按[后端契约](../../daso/docs/external_backends.md)和[复现流程](../../daso/docs/reproduction.md)配置独立环境。归档环境记录包含 PyTorch 2.9.0+cu128、Transformers 4.55.2；phase runtime 对应 vLLM 0.11.1，任务评测使用 lm-eval 0.4.12。实际安装版本仍以各 run 的环境记录和新机器兼容性检查为准，不能将自动安装的新版依赖视为相同数值环境。

[依赖清单](portability/dependencies/manifest.json)记录了实际 fork、wrapper、上游 vLLM、SmoothQuant、MIRROR、FakeVLM 的基线及本地源码改动。后四者的附带快照用于保全上下文，不能全部自动应用到实际 phase runtime。尤其 upstream_vllm 的依赖删减补丁属于旧本地环境，默认不需要套用。

## 推荐代码入口

| 内容 | 入口 | 使用注意 |
| --- | --- | --- |
| 可配置的 DASO 核心、建模和求解 | [daso](../../daso/README.md)、[复现文档](../../daso/docs/reproduction.md) | 适合作为新模型流程起点，显式传模型与后端路径 |
| 当前完整同轮主表执行 | [048](runs/048/README.md) | 原容量 arena/Graph，适合检查方法与计时语义 |
| 邻近双场景数值代码 | [049](runs/049/README.md) | 实际 KV teacher forcing、Graph、8bit与隔离兼容头 |
| 最终空闲窗口速度 | [053](runs/053/README.md) | 代码与原始结果可参考；九小时截止已过，不直接重启旧调度器 |
| 原协议完整任务精度 | [050](responses/reviewer2/qwen_original_quality_run050/README.md) | 生成上限256、正常EOS；与固定测速长度区分 |
| HAWQ 与 DASO 同候选对照 | [041](responses/reviewer2/llama31_matched_allocator_run041/README.md) | 避免混淆质量预算和延迟上限 |
| mixed 调度补充 | [047](responses/reviewer2/mixed_pd_compatibility/README.md) | 双表示常驻 eager，与单 arena 结果分开 |

旧 run 中的路径是历史记录，很多脚本硬编码了原模型路径、vLLM路径、CPU绑定和绝对截止时间。不要对历史代码批量替换路径或改写原始 provenance；在新 run 复制必要入口、增加显式配置，并记录新源码和依赖 SHA。

原 Qwen3-8B 模型路径为 /root/wja/data/models/Qwen/Qwen3-8B；新机器需自行提供目标模型。tokenizer及checkpoint导出也需从该机器模型重建。已有校准/测试 token 小文件保留，原模型权重与 canonical SparseGPT 状态不随 Git 传输；后者通过原脚本重建并记录数值身份。

## 原始结果的无损归档

可读的代码、Markdown、CSV、配置、较小 JSON 和实验日志直接进入 Git。大量逐窗 capture、phase trace、较大 JSON 和 profiler 原始文件存为分组压缩归档，单个分片不超过40 MiB；本地原文件没有删除。

055新增归档10,122个原始文件，约715.2MiB，18个分片均不超过40MiB；原文件保留，分片及逐文件SHA恢复核验通过。

列出可恢复组：

~~~bash
python artifacts/rebuttal/portability/restore_raw_results.py
~~~

按需恢复一个组，例如完整任务结果或预算审计：

~~~bash
python artifacts/rebuttal/portability/restore_raw_results.py --group run-055
python artifacts/rebuttal/portability/restore_raw_results.py --group run-050
python artifacts/rebuttal/portability/restore_raw_results.py --group run-003
~~~

需要完整原始证据时：

~~~bash
python artifacts/rebuttal/portability/restore_raw_results.py --all
~~~

脚本检查分片 SHA、逐文件 SHA，并在已有文件内容不同的情况下停止，不覆盖不同结果。[分组清单](portability/raw_archives/manifest.json)与各组逐文件清单记录原路径和字节。部分完整来源审计需要先恢复对应组；不能仅凭缺少默认展开的 trace 认定原实验缺失。

同仓库内可解析的历史绝对 symlink 改为指向相同文件的相对链接，内容未变；一个明确标记为历史 dangling 的参考链接保留原状。迁移记录见[symlink清单](portability/symlinks.json)。可重建下载缓存、编译缓存、PID/锁文件与模型权重未提交。

## 已有结论与不能混用的口径

- **R1预算：** 原 DP 七候选扩展，9负载×2预算，18设定，每点100窗，17/18均值满足；唯一超限为0.001959 nats/token。复用16策略和共同文本，不是18次独立试验。
- **原三模型主表：** [Qwen](responses/reviewer2/qwen3_main_table/README.md)、[Llama-3.1](responses/reviewer2/llama31_main_table/README.md)、[Llama-2](responses/reviewer2/llama2_main_table/README.md)。HAWQ为FP4/INT8/BF16后端的PTQ适配，两场景独立选点。
- **最新多配置：** 053在GPU3完成三轮，含三种Prefill、四种P+D及输出256/512。扩容后 P+D 排名有变化，不能用048优势替代053数据；†波动标记保留。
- **质量口径：** 原任务 Avg 可按策略/batch复用；固定 S/O 的 NLL另测。长Decode专门NLL尚未完成，不把原Avg改标签当O512新精度。
- **内存：** 048为单arena/Graph，047为双表示常驻eager；GPU峰值与持久Host缓存分列。
- **求解：** 精确求解针对代理，不保证真实NLL/SLA；“几秒”是已有画像和校准后的求解时间，完整准备成本另计。
- **未完成：** 持续错峰生产负载、双worker PD交接；055完整32B仍为单卡固定批协议，不能据此声称生产在线服务或多卡部署已验证。

## 可直接给新机器 Codex 的提示

> 先阅读根AGENTS.md、本交接、中文总稿R4第2点及055完整报告。055的完整Qwen3-32B三方法对照、100窗test、三轮同卡速度和61个完整任务/分片已完成并终审通过，不要重复启动055或旧053调度器。原始证据可按run-055恢复并核验SHA。后续新实验使用未占用编号，先核对真实设备、模型、fork补丁和kernel环境，按新模型/硬件重做质量校准及延迟画像。持续错峰生产服务和双worker PD仍未验证。不要删除代码、日志、CSV、文档或原始结果，不改标签冒充新测量，不因胜负筛掉已完成轮次。
