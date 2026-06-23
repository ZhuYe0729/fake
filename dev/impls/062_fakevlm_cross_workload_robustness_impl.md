## 2026-06-23 - Initial cross-workload scaffold
- 开发目的：为 FakeVLM 建立 prefill-only、normal_01、normal_02 三个 workload 下的 uniform 与 linear hybrid 速度对比流程。
- 修改内容：新增计划文件和 debug artifact 脚本目录；实现 E2E 速度测试、4-GPU 任务级 launcher、summary 表生成入口。
- 影响文件：`dev/plans/062_fakevlm_cross_workload_robustness_plan.md`，`dev/impls/062_fakevlm_cross_workload_robustness_impl.md`，`artifacts/debug/026_fakevlm_cross_workload_robustness/`。
- 后续注意：完整速度数据需要在 GPU 0-3 上运行 full launcher；同一 GPU 不应并发多个速度任务。

## 2026-06-23 - Normal workload smoke fix
- 开发目的：修复 normal_01/normal_02 长输入测速时 finite 检查额外占用显存导致的 OOM。
- 修改内容：将 E2E runner 的 logits finite 检查从完整序列改为只检查最后一个 token logits，避免为 `batch=1,input=16384` 的完整 logits 额外分配 FP32 缓冲。
- 影响文件：`artifacts/debug/026_fakevlm_cross_workload_robustness/scripts/run_e2e_speed.py`，`dev/impls/062_fakevlm_cross_workload_robustness_impl.md`。
- 后续注意：测速 forward 路径不变；该检查仅用于发现非有限输出。
