## 2026-10-27 — 冻结描述性统计口径
- 固定5000次paired bootstrap和seed1632026，保持原ROUGE-L%/corpus BLEU定义及配对输入，既有任务分数不改。
- Qwen先运行；Llama等待完整输出及037复核，若基线重放不一致先诊断，不直接以旧分数宣称优势。CPU绑定NUMA1相关核心，数值库单线程。

## 2026-10-27 - Qwen区间已完成并复核
- 16项比较、每项5000次配对重采样，原分数聚合与来源/交付SHA均校验通过，PNG/PDF已查看。
- 两个点相对INT8的Gen Avg差分别+0.29412（95%描述区间0.06656–0.52982）、+0.23291（0.02027–0.45253）；相对SQ也为正，个别分项仍跨0。
- 仅固定配置条件下的描述统计，未校正选点/多重比较，不能称为新盲测或确证性胜出。输出responses/reviewer2/quality_uncertainty_night161/qwen3。
- 图直接画百分位区间和观察值，避免极端情况下把区间裁到观察值；CSV不变。

## 2026-10-27 - CPU统计与正式计时互斥
- 后续bootstrap脚本持全局measurement SH锁，正式测速EX锁会与其互斥；CPU密集ROUGE/重采样不会在正式计时期间争用宿主资源。质量评测可并行，数值方法/seed/样本不改。
- 已完成Qwen使用的原脚本按其provenance.code_sha留为qwen3/source.py；后续Llama也随交付保留执行脚本，不覆盖Qwen已有CSV或区间。
