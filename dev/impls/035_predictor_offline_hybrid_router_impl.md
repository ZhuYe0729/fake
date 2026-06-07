## 2026-06-05 - Predictor offline hybrid router
- 开发目的：基于 kernel latency predictor 实现通用离线 hybrid kernel 策略选择，并接入 Qwen3.5 checkpoint/benchmark 流程。
- 修改内容：
  - 新增 `fake/kernels/offline_hybrid_policy.py`，提供通用 Linear shape + scenario 到 per-layer policy 的选择 API。
  - 策略枚举同 kernel 与 `dense_nvfp4`/`marlin_nvfp4` 兼容双后端组合，并计入 NVFP4 canonical-to-backend 转换预测成本。
  - 新增 `predictor_hybrid` Qwen3.5 method，支持从 policy JSON 构建手动 hybrid 或共享 NVFP4 hybrid 模块。
  - 新增 `scripts/analyze_offline_hybrid_policy.py`，用于从通用 shapes JSON 或 Qwen3.5 meta 模型生成 policy JSON/CSV。
  - 更新 Qwen checkpoint 和 E2E benchmark 脚本，支持 `--policy-json`。
- 影响文件：
  - `fake/kernels/offline_hybrid_policy.py`
  - `fake/models/qwen3_5_kernels.py`
  - `scripts/analyze_offline_hybrid_policy.py`
  - `scripts/prepare_qwen3_5_kernel_checkpoint.py`
  - `scripts/bench_qwen3_5_swh_e2e.py`
- 后续注意：真实 GPU checkpoint 构建与 E2E 延迟验证需要在计算节点执行；当前登录节点只完成无 GPU 的策略生成和 CPU smoke 检查。

## 2026-06-05 - 三模型 predictor hybrid 对比
- 开发目的：使用 kernel latency predictor 为 Llama-2-7B、Llama-3.1-8B、Qwen3.5-9B 在 prefill-only 与 normal 场景下生成离线 hybrid 策略，并与已有手动策略结果对比。
- 修改内容：
  - 修正 `output_tokens=0` 的 prefill-only 策略选择，不再要求 decode shape 支持。
  - 新增 `scripts/analyze_predictor_hybrid_vs_manual.py`，读取已有手动 benchmark artifacts，生成 predictor policy、策略差异表和延迟汇总。
  - 输出结果到 `artifacts/results/benchmarks/hybrid/pred/`。
- 影响文件：
  - `fake/kernels/offline_hybrid_policy.py`
  - `scripts/analyze_predictor_hybrid_vs_manual.py`
  - `artifacts/results/benchmarks/hybrid/pred/*`
- 后续注意：本次输出是 predictor 离线线性层端到端估算与已有手动真实/模块结果的并列表；当前 `nvidia-smi` 无法连接 NVIDIA driver，未重新跑全模型真实 GPU E2E。

## 2026-06-05 - Predictor policy GPU module timing
- 开发目的：在可见 GPU 环境下，对 predictor 生成的三模型两场景策略做真实 kernel module 计时。
- 修改内容：
  - 新增 `scripts/bench_predictor_hybrid_policy_modules.py`，按 policy 为每个 linear group 构建选中 kernel，并分别计时 prefill/decode。
  - 生成 `gpu_policy_module_e2e.csv` 和 `gpu_policy_module_summary.csv`。
  - 更新 `artifacts/results/benchmarks/hybrid/pred/README.md`，加入 GPU module E2E 检查表。
- 影响文件：
  - `scripts/bench_predictor_hybrid_policy_modules.py`
  - `artifacts/results/benchmarks/hybrid/pred/gpu_policy_module_e2e.csv`
  - `artifacts/results/benchmarks/hybrid/pred/gpu_policy_module_summary.csv`
  - `artifacts/results/benchmarks/hybrid/pred/README.md`
- 后续注意：这是真实 GPU kernel/module 计时并按层数汇总，不是完整 Transformer forward；完整全模型 E2E 仍需要为 Llama 接入 predictor policy 的全模型替换路径。
