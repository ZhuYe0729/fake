# 117 Llama2 B=8/O=64 prefill-decode implementation

## 2026-07-19 - Scenario bootstrap and runtime gate
- 开发目的：为 B=8/S=2048/O=64、BF16-KV 的 canonical prefill-decode 闭环建立隔离实验目录，并在大规模校准前阻断旧 runtime 与 sparse 导出问题。
- 修改内容：创建 056 实验配置、从 055 复制策略覆盖集并重定位 manifest；派生固定 WikiText 2048+64 样本；重审 kernel actions（Mpre=16384、Mdecode=8）；新增 canonical checkpoint provenance 校验和 B=8/O=64 fresh-process speed runner。扩展 NLL launcher 以显式传入 input/output token 长度。
- 验证：dense-BF16 与含 sparse-NVFP4 的 p03 均使用 canonical sparse states、`prune:false`。两者的 vLLM V1 trace 都只有 `prefill/tokens=16384` 128 次和 `decode/tokens=8` 8064 次；dense KV capacity 为 22,672 tokens，满足单 wave。
- 后续注意：decode M=8 下 sparse-NVFP4 action 因 `M%32!=0` 被 audit 排除。已完成 8 个 canonical phase-local error 表；待启动 O=64 全量 teacher-forced NLL labels 与 speed calibration。

## 2026-07-19 - Fresh-process speed calibration completed
- 开发目的：以正式 B=8/S=2048/O=64、BF16-KV、关闭 chunked prefill 的 vLLM 协议，对 roofline kernel-cost 聚合进行 E2E 校准。
- 修改内容：完成 12 个覆盖策略的独立进程测量；每个策略包含 TTFT 与 E2E 各 1 次 warmup、5 次正式重复。写入 `speed/runs/*/iterations.csv`、`speed/calibration/calibration.csv` 和 `metrics.json`。
- 验证：12/12 策略完整完成，无 OOM 或运行错误。E2E 正式样本的 CV 大多低于 2.1%；p18 有一次 3567 ms 的瞬态离群值（median 2317 ms），p21 有一次 2969 ms 的较高值，均以五次中位数进入校准。
- 后续注意：现有“单一 raw E2E cost -> 单调映射”不适合作为最终速度模型：holdout 的 dense-anchor scale MAE 为 178.9 ms，单调映射 MAE 为 286.3 ms。下一步应保留 roofline 基模型，补充阶段/方法感知的自然校准项，而不是直接用该单调映射求解。

## 2026-07-19 - Same-runtime uniform speed baselines fixed
- 开发目的：在进入 NLL 建模前固定正式 uniform 速度参照，并避免再次混用 uniform 专用 runtime 与 phase-hetero runtime。
- 修改内容：将 p00--p04 的既有正式测速汇总为 `speed/uniform_baselines.{csv,md}`；求解器删除不可靠的全局 E2E 单调映射，只保留原有 per-kernel roofline+residual 成本和及其 B=8/O=64 阶段公式。
- 验证：五行均来自相同的 VLLM V1 `phase_hetero_mytest` protocol（BF16 KV、关闭 chunked prefill、B=8/S=2048/O=64、fresh process、5 次正式重复）。
- 后续注意：decode M=8 不支持 sparse-NVFP4，故该 baseline 必须标为 legal projection（prefill sparse-NVFP4 / decode dense-NVFP4），不能声称为全阶段 uniform sparse-NVFP4。

## 2026-07-19 - B=8 teacher-forcing capacity correction
- 开发目的：使实际 vLLM NLL 标签的调度容量与 B=8/S=2048/O=64 的正式场景一致。
- 修改内容：NLL runner 改为显式记录并使用 `gpu_memory_utilization=0.8`；将 `max_num_batched_tokens` 从仅 prefill 的 `B*S=16384` 修正为完整 wave 的 `B*(S+O)=16896`；capture 额外记录 callback 的 output length，便于审计每步 teacher forcing。
- 验证：0.7 配额只有约 16,464 KV tokens，小于完整 8-request wave 的 16,896，因而第八个请求出现 `t0,t0,t1,...` 的重调度伪象。p06 在 0.8 下的 8 个请求均精确捕获 64 个预期 target token（包括最后一个请求），phase trace 也完整进入 decode。
- 后续注意：此前 B=1 或 0.7 的 NLL 结果不可混入正式 B=8 质量拟合；后续全量任务统一采用 0.8，并在多 GPU 启动时错峰以降低初始化瞬时负载。

## 2026-07-19 - Candidate speed closure
- 开发目的：对质量模型求解得到的 10 个合法候选，用与 uniform baseline 相同的 fresh-process vLLM runtime 完成实际 E2E 速度闭环。
- 修改内容：新增可恢复的候选速度 dispatcher 和 closure 汇总；每点保留 canonical checkpoint provenance，执行 1 warmup + 5 measured repeats（B=8/S=2048/O=64、BF16 KV、关闭 chunked prefill）。
- 验证：10/10 完成。中位数实测 speedup 为 1.000、0.978、0.953、1.040、1.148、1.189、1.316、1.442、1.480、1.561；高加速端与 roofline 筛选的绝对数值有差异，故论文只应使用 closure 中的实测值。
- 后续注意：个别策略出现单次瞬态慢值，仍以 5 次中位数为正式值；p01/p02 实测慢于 dense BF16，保留在速度表但不进入耗时的下游任务验证。

## 2026-07-19 - Downstream task closure and sparse-OOM recovery
- 开发目的：在实测速度有效的 8 个候选上完成 CNN/DM、DialogSum、IWSLT 的真实 vLLM 生成验证，并生成论文可用的速度—任务质量图。
- 修改内容：复用 speed closure 的 canonical checkpoint；完成 22,664 个样本。三个长 prompt shard 因 sparse-BF16 workspace 与 VLLM KV cache 竞争而在 batch=4 下 OOM；保留已有输出，使用不重叠的 retry shard 补齐缺失 question id，batch=1，最后 3 条极长样本将 KV utilization 降至 0.50。新增该恢复脚本及候选速度汇总。
- 验证：所有 24 个 policy×dataset 组合均达到目标样本数且 question id 无重复；24/24 指标文件生成。报告使用同 runtime 的 5-repeat median speed：b8o64003=1.040x、b8o64004=1.148x、b8o64009=1.561x。生成 `task_quality/report/summary.md` 和三个任务 Pareto 图。
- 后续注意：下游任务的 B=1 retry 仅用于避免输入长度导致的 workspace OOM，不改变权重、policy 或生成配置；速度声明仍只使用正式 B=8 fresh-process speed closure。
