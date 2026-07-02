## 2026-06-23 - FakeVLM linear time proportion scaffold
- 开发目的: 新增 FakeVLM dense BF16 的 `nn.Linear` 时间占比测试和分析，和已有 Llama2/Qwen 线性占比实验形成对照。
- 修改内容: 新增 028 debug 实验目录、4GPU launcher、单 workload benchmark、summary 脚本和 README；新增 064 plan。
- 影响文件: `artifacts/debug/028_fakevlm_linear_time_proportion/*`, `dev/plans/064_fakevlm_linear_time_proportion_plan.md`, `dev/impls/064_fakevlm_linear_time_proportion_impl.md`
- 后续注意: 默认读取 FakeClue 测试集和 `/home/agent/wja/data/models/lingcco/fakeVLM`；完整运行依赖本机 GPU 和 `cospaq` 环境。

## 2026-06-23 - Smoke validation and launcher hardening
- 开发目的: 验证 FakeVLM 模型/数据路径、forward、hook 计时和 summary 生成可用。
- 修改内容: 通过 Python/shell 静态检查；完成 `prefill_b1_i1024` 的 1-iter smoke；launcher 切换到 repo root 后运行；breakdown 只测首个 decode step，避免长 decode workload 重复消耗。
- 影响文件: `artifacts/debug/028_fakevlm_linear_time_proportion/results/`, `artifacts/debug/028_fakevlm_linear_time_proportion/summary/`, `artifacts/debug/028_fakevlm_linear_time_proportion/scripts/*`
- 后续注意: smoke 显示 FakeVLM 有 371 个 Linear，其中 language=224、vision=144、projector=2；全量运行会覆盖同 workload 的 smoke 行。

## 2026-06-23 - Full 4GPU run completed
- 开发目的: 生成 FakeVLM linear 时间占比完整默认 workload 结果。
- 修改内容: 使用 GPU 7/6/5/4 跑完 6 个 workload，并生成 raw CSV、summary CSV、analysis report。
- 影响文件: `artifacts/debug/028_fakevlm_linear_time_proportion/results/fakevlm_linear_proportion_raw.csv`, `artifacts/debug/028_fakevlm_linear_time_proportion/summary/*`, `artifacts/debug/028_fakevlm_linear_time_proportion/logs/*`, `artifacts/debug/028_fakevlm_linear_time_proportion/status/*`
- 后续注意: 默认结果显示 1024-token prefill linear 占比约 73-79%，4096-token prefill 降到约 58%，16384-token decode workload 的 prefill linear 约 31%、decode linear 约 16%。
