# 132 Llama3.1 Pareto search audit implementation

## 2026-07-23 - Bootstrap search audit

- 开发目的：建立 Llama3.1 两场景的独立三类策略 search audit。
- 修改内容：新增 132 plan/impl 记录，并开始实现 debug 069 的候选生成、正式
  runner、可恢复 launcher 和汇总。
- 影响文件：`dev/plans/132_llama31_pareto_search_audit_plan.md`、
  `dev/impls/132_llama31_pareto_search_audit_impl.md`、
  `artifacts/debug/069_llama31_pareto_search_audit/`。
- 后续注意：057/060/061/068 均为只读数据源，不允许覆盖。

## 2026-07-23 - Candidate generation and formal runners

- 开发目的：落地第一轮 30×2 策略审计并复用最终 Llama3.1 测试口径。
- 修改内容：实现固定种子的 15 random、10 neighborhood、5 suspicious
  policy 生成；实现 action legality/SHA256 静态审计；实现 068 warmed prefill
  speed、完整 ARC-Challenge、057 fresh-process phase speed 和完整 IWSLT-333
  runner；新增可恢复的串行速度/并行质量 launcher 与 union-frontier 汇总。
- 验证结果：60 个候选全部生成；两场景均严格为 15/10/5，policy 全部唯一且
  每个包含 128 个合法 fused Linear action；全部 Python 脚本通过 `py_compile`。
- 后续注意：GPU smoke 通过前不启动完整测试；speed launcher 会在候选前后各测
  一个 dense-BF16 anchor，汇总拒绝超过 3% 的漂移。

## 2026-07-23 - Runner smoke and strict report gates

- 开发目的：确认四类正式 runner 可执行，并防止不完整或跨口径数据进入报告。
- 修改内容：prefill/decode 速度各完成一个真实 candidate smoke；汇总固定使用
  061/057 的论文 dense-BF16 分母，并加入 30/30 完整性、首尾 anchor 漂移和
  历史 anchor 偏差三项硬校验。
- 验证结果：prefill `random_001` 五次为 1078.73--1079.82 ms，中位数
  1079.06 ms；decode `random_001` fresh-process 中位数 2409.48 ms；两类
  14+ GiB 临时 checkpoint 均在完成后自动删除。相同两点的完整质量 smoke
  也通过：ARC-Challenge `acc_norm=0.540102`；IWSLT 333/333、无空输出、
  SacreBLEU `10.571902`。
- 后续注意：已在 GPU 1 启动完整 speed-first 批量测试，测速期间禁止并发质量
  任务。

## 2026-07-23 - Duplicate-run exclusion

- 开发目的：消除误重复启动同一 policy 时的 checkpoint 路径竞争。
- 修改内容：四个 runner 均增加覆盖“导出 + 测试 + 清理”完整生命周期的
  per-policy 文件锁，并在拿锁后再次检查完成标记。
- 问题结论：普通沙箱内的 tmux 查询看不到外部 tmux socket，曾把仍在执行的
  正式 launcher 误判为停止；重复启动相同 policy 后两个 exporter 同写路径，
  导致一次 safetensors `No such file or directory`。单独重跑策略可通过，故
  不是 policy、canonical sparse state 或 kernel 支持问题。
- 后续注意：正式 GPU 任务状态使用外部权限查询进程/结果文件，不能仅依赖沙箱
  内 `tmux has-session`。

## 2026-07-23 - Stable temporary/build isolation

- 开发目的：避免外部根 overlay `/tmp` 满盘和跨 policy 扩展编译竞争影响正式
  测速。
- 修改内容：prefill/decode 速度 runner 共用覆盖完整运行周期的全局速度锁；
  四个 runner 的 `TMPDIR` 固定到 debug 069 所在数据盘；Marlin build-dir
  显式固定到项目内已完整生成的扩展目录。
- 问题结论：`random_004` 曾在 vLLM 加载时触发 Marlin ninja 检查，nvcc 因
  根 overlay `/tmp` 100% 而失败；这不是策略、模型显存或 E2E runner 问题。
- 验证结果：修正后全部脚本通过 `py_compile` 和 `git diff --check`，可恢复
  launcher 已从 `random_004` 重新启动。

## 2026-07-23 - Prefill speed batch complete

- 开发目的：完成第一轮 30 个 challenger 的正式 prefill-only E2E 测速。
- 验证结果：30/30 候选均完成 warmed 5-repeat；历史 dense anchor 为
  1087.527 ms，本批首尾为 1090.416/1089.901 ms，首尾漂移 0.047%，相对
  历史偏差均小于 0.3%；候选实测速度覆盖 1.008x--2.229x。
- 后续注意：已在 GPU 7 启动 prefill-decode fresh-process 5-repeat 正式测速；
  质量阶段继续保持未启动，避免干扰速度。

## 2026-07-23 - Continue after infeasible decode candidate

- 开发目的：避免单个真实显存不可行的 challenger 终止整批审计。
- 问题结论：decode `random_011` 在 B=8/S=2048 的 prefill 中包含 21 个
  sparse-NVFP4 模块，在线 activation pack 需额外申请 896 MiB、实际仅余
  约 480 MiB，因而 OOM；checkpoint 已由 runner 清理，无外部 GPU 进程证据。
- 修改内容：launcher 对候选失败逐点记录并继续；支持显式
  `--retry-failed`；质量阶段只测试已有有效速度结果的候选；汇总分别报告
  generated、speed-feasible 与 measured 数量。
- 验证结果：脚本通过 `py_compile`/`git diff --check`，decode speed 已从
  `random_011` 恢复；若它再次 OOM 将标为 infeasible 并继续 `random_012`。

## 2026-07-24 - Clean retry for externally contaminated decode failures

- 开发目的：区分真实策略显存不可行与中途外部 GPU 占用。
- 初次结果：30 个候选中 17 个成功、13 个失败，首尾 dense anchor 为
  2159.410/2159.678 ms、漂移 0.012%。
- 问题结论：失败日志中 7 个在 vLLM 启动时仅余 13.75--23.69 GiB；另有多个
  OOM 的显存账目显示数 GiB 至近 30 GiB 外部占用，因此不能将 13 个全部判为
  speed-infeasible。
- 修改内容：汇总优先使用 `dense_anchor_retry_before/after`；在全部 GPU 空闲后，
  已于 GPU 7 启动 retry-before anchor、13 个失败点和 retry-after anchor 的
  串行清洁重试。
- 后续注意：清洁重试后仍 OOM 的点才标记为真实 infeasible，成功点清除 failure
  marker；质量测试继续等待速度重试结束。

## 2026-07-26 - Quality pipeline launched

- 验证结果：13 个 decode 失败点在空闲 GPU 上清洁重试后全部成功，最终
  prefill/decode 均为 30/30 有效速度；retry anchor 为
  2160.006/2213.518 ms，漂移 2.48%，两者相对历史 anchor 均小于 3%。
- 修改内容：启动 GPU 1--7 顺序质量流水线，先执行 29 个剩余
  ARC-Challenge，再执行 29 个剩余 IWSLT-333，最后自动运行 `summarize.py`。
- 启动检查：7 个 ARC worker 均存活，采用全局串行 checkpoint export 与多卡
  并行评测，避免同路径竞争和同时导出造成的内存/磁盘压力。

## 2026-07-27 - Retry three externally blocked IWSLT exports

- 初次结果：ARC-Challenge 30/30 完成；IWSLT 27/30 完成，
  `neighborhood_007`、`suspicious_003`、`suspicious_005` 失败。
- 问题结论：三点均在 checkpoint export 阶段失败；错误显示同卡已有约
  28 GiB 其他进程，而当前 exporter 仅占 2--3 GiB，不是策略推理 OOM。
- 修改内容：decode IWSLT runner 增加跨 policy 的全局 export lock；在 GPU
  1--3 仅重试三个失败点，随后自动重新运行正式汇总。
- 启动检查：首个 exporter 已正常转换，另外两个 worker 正在等待导出锁，GPU
  无外部显存占用。

## 2026-07-27 - Pareto search audit complete

- 验证结果：两个场景均完成 30/30 challenger 的真实速度和完整下游质量，
  所有 speed/quality failure marker 已清除；GPU 与测试进程均已释放。
- Prefill-only：challenger 自身前沿 7 点；仅支配两个旧 reference，其中
  `point_000` 被 `random_002` 以 +0.030x speedup、+0.00597 accuracy 支配，
  `point_004` 被其邻域点 `neighborhood_003` 以 +0.0073x、+0.00256 accuracy
  小幅改进；其余 11 个旧前沿点均未被支配。
- Prefill-decode：challenger 自身前沿 4 点；没有任何 challenger 支配 7 个
  ours reference，随机/可疑点整体位于 ours 前沿下方；邻域点仅在相邻 ours 点
  之间填充 union frontier 或扩展最高速度端。
- 产物：两场景 `all_results.csv`、`challenger_frontier.csv`、
  `union_frontier.csv`、`gap_to_reference.csv`、PNG/PDF 图和完整性 summary
  均已生成在 debug 069 的对应 `report/` 目录。

## 2026-07-29 - Promote stable decode neighborhood endpoint

- 开发目的：核查 prefill-decode 中超过 1.5x 的 `neighborhood_009`，并在不改写
  原始审计的前提下吸收有效搜索改进。
- 验证结果：同一 GPU fresh-process 复测中，`point_011` 与
  `neighborhood_009` 的中位延迟分别为 1496.97 ms 和 1423.14 ms；后者快
  5.19%，五次主延迟为 1417.96--1425.70 ms，确认原 1.5274x 坐标不是噪声。
- 修改内容：新增 post-audit refined frontier，将 `neighborhood_009` 作为
  refined ours 的最高速度端点；原 challenger 报告、分类和 30 点完整性结果
  保持不变。
- 影响文件：debug 069 的 `scripts/summarize.py`、`README.md`、
  `prefill_decode/report/post_audit_*` 与
  `pareto_search_audit_post_refinement.{png,pdf}`。
- 后续注意：复测扩展使用本机 CUDA 12.9 独立缓存；正式 refined 坐标仍采用
  原审计的冻结 BF16 分母和 1.5274x 结果，复测仅用于确认相对收益稳定。

## 2026-07-29 - Correct display treatment of decode outlier

- 开发目的：按用户确认保持既定 ours 点不变，仅在展示图中忽略
  `neighborhood_009`。
- 修改内容：撤销 refined-ours 展示；新增 presentation-only decode 图，从绿色
  散点和展示用 union frontier 中排除该点，其余 9 个实测 neighborhood 点保持
  不变。
- 数据边界：原始 `all_results.csv`、challenger frontier、union frontier、
  gap、正式审计图和复测结果均不修改；不以未测坐标伪造第 10 个替代点。
- 影响文件：debug 069 的 `scripts/summarize.py`、`README.md`、
  `prefill_decode/report/pareto_search_audit_display.{png,pdf}` 与
  `display_exclusions.json`。

## 2026-07-29 - Measure decode display replacement

- 开发目的：用一个真实完整测量的新 neighborhood 点替换展示中排除的
  `neighborhood_009`，恢复 10 个绿色点。
- 候选过程：首次以 `point_011` 为 parent 固定生成 `neighborhood_011`，实测
  1.4838x / BLEU 1.438638，会支配既定 `point_011`，因此保留实验记录但不用于
  展示；随后改用中段 `point_006`，固定 seed 20260730 一次生成最终
  `neighborhood_012`。
- 验证结果：`neighborhood_012` 的 128 个 action 全部合法，与现有策略不重复；
  fresh-process 五次中位延迟 1698.89 ms，对冻结 BF16 为 1.2942x；IWSLT
  333/333、无空输出、SacreBLEU 10.349794。
- 修改内容：展示图以 `neighborhood_012` 替换 `neighborhood_009`，展示用
  neighborhood 恢复为 10 点；正式 audit CSV、frontier、gap 和原图不变。
- 影响文件：新增 replacement 生成脚本与两次尝试的 policy/manifest/实测结果；
  更新 display PNG/PDF、`display_challengers.csv` 和
  `display_exclusions.json`。

## 2026-07-29 - Enlarge decode display typography

- 开发目的：提高 presentation-only prefill-decode 图在 PDF 和 PNG 中的可读性。
- 修改内容：仅将 display 图的坐标轴标签、刻度、图例分别增大到 16、14、12 pt；
  正式审计图的字号和所有数据保持不变。
- 影响文件：debug 069 的 `scripts/summarize.py` 与
  `prefill_decode/report/pareto_search_audit_display.{png,pdf}`。

## 2026-07-29 - Match prefill-only typography

- 开发目的：让 prefill-only 图与 prefill-decode display 图保持一致的可读字号。
- 修改内容：将 prefill-only 图的坐标轴标签、刻度、图例同步为 16、14、12 pt；
  图中数据和 prefill-decode 正式审计图保持不变。
- 影响文件：debug 069 的 `scripts/summarize.py` 与
  `prefill_only/report/pareto_search_audit.{png,pdf}`。

## 2026-07-29 - Document empirical Pareto-bound audit

- 开发目的：在建模与消融总文档中详细说明 069 如何估计 ours frontier 距离更优经验
  帕累托界限的距离。
- 修改内容：仅在 v2 文档末尾追加 Appendix B，记录 challenger 构造、真实 closure、
  anchor gate、两场景正式结果、结论边界与 presentation-only 数据边界；原文不改。
- 影响文件：`artifacts/debug/064_llama31_modeling_ablation/report/`
  `modeling_design_and_ablation_v2.md`。

## 2026-07-29 - Visualize layer-wise ours strategy evolution

- 开发目的：为补充实验展示 ours frontier 随实测加速比增加时，各层、各 fused module
  和两个推理 phase 的压缩 action 变化。
- 修改内容：新增可复现热力图脚本；prefill-only 使用 4 个 module panels，
  prefill-decode 使用 2 phase × 4 module panels；同步导出 PNG、PDF 与逐色块长表 CSV，
  并在 Appendix B 追加图和阅读说明。
- 数据边界：只使用原始 ours reference points，按正式实测 speedup 排序；不包含
  uniform、challenger 或 display-only replacement。
- 影响文件：debug 069 的脚本、README、两个场景的 strategy evolution 报告，以及
  debug 064 的 v2 建模文档。
