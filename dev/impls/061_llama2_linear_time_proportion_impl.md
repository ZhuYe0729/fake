## 2026-06-23 - Llama2-7B linear time proportion scaffold
- 开发目的: 在 022 线性层用时占比实验中新增 Llama2-7B 专属测试，不覆盖已有 Qwen 结果。
- 修改内容: 新增 `llama2_7b` 实验目录、并行 launcher、benchmark 脚本、分析脚本和说明文档；新增 061 plan 记录。
- 影响文件: `artifacts/debug/022_linear_time_proportion_study/llama2_7b/*`, `dev/plans/061_llama2_linear_time_proportion_plan.md`, `dev/impls/061_llama2_linear_time_proportion_impl.md`
- 后续注意: 完整运行依赖本机 GPU 7/6/5/4、Llama2-7B 本地权重和 conda 环境 `cospaq`。

## 2026-06-23 - Static checks and launch attempt
- 开发目的: 验证新增脚本可解析，并按计划启动本机多 GPU 测试。
- 修改内容: 通过 `py_compile` 和 `bash -n` 静态检查；执行 `run_parallel.sh` 启动测试。
- 影响文件: `artifacts/debug/022_linear_time_proportion_study/llama2_7b/logs/speed_gpu*_shard*.log`
- 后续注意: 当前沙箱 CUDA 不可用，worker 在 `torch.cuda.is_available()` 处失败；目标 GPU 机器上可直接重新运行 `bash artifacts/debug/022_linear_time_proportion_study/llama2_7b/run_parallel.sh`。

## 2026-06-23 - Host GPU launch
- 开发目的: 使用宿主 GPU 权限绕过沙箱 CUDA 限制，实际启动 Llama2-7B 多 GPU 测试。
- 修改内容: 以 `cospaq` 环境启动 `run_parallel.sh`；GPU 7/6/5/4 均成功加载 Llama2-7B 并进入 speed shard 循环。
- 影响文件: `artifacts/debug/022_linear_time_proportion_study/llama2_7b/speed/`, `artifacts/debug/022_linear_time_proportion_study/llama2_7b/logs/`
- 后续注意: speed 阶段由 output=256 的 shard 主导耗时；完成后 launcher 会自动进入 breakdown 阶段并运行 `analyze.py`。
