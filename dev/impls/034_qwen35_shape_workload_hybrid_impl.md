## 2026-06-02 - Dense NVFP4 same-weight hybrid
- 开发目的：实现 Plan 034 的第 0 部分，让 Qwen3.5 的 `W4A16 marlin_nvfp4` 与 `W4A4 dense_nvfp4` 共享同一份 dense NVFP4 canonical 权重并按输入负载切换。
- 修改内容：新增 `hybrid_nvfp4` method；prepare 阶段生成 `qwen3_5_hybrid_nvfp4_packed_v1` checkpoint；runtime 使用 `QwenHybridDenseNVFP4Linear` 在 `M<=16` 走 Marlin W4A16，否则走 CUTLASS dense W4A4；sparse 路径保持独立。
- 影响文件：`dev/plans/034_qwen35_shape_workload_hybrid_plan.md`、`fake/models/qwen3_5_kernels.py`。
- 后续注意：当前会话未执行 GPU prepare/forward；后续需要在 RTX 5090 上 prepare `hybrid_nvfp4` checkpoint 并做 prefill/decode smoke。

## 2026-06-02 - Qwen3.5 2B 手动 layer-level hybrid
- 开发目的：基于 `5_kernel_comprehensive` 的 layer-level GEMM 结果，为 Qwen3.5-2B 的 5 个典型场景提供手动 hybrid scheme。
- 修改内容：新增 `manual_hybrid_m1/m4/m8/m16` 四个 method；prepare 阶段按模块后缀分别构建 BF16、dense NVFP4、sparse BF16、sparse NVFP4、Marlin backend；runtime 按 `M` 在 prefill/decode backend 间切换。
- 影响文件：`fake/models/qwen3_5_kernels.py`、`dev/impls/034_qwen35_shape_workload_hybrid_impl.md`。
- 后续注意：当前仅完成 py_compile 和 CLI help 检查，未在 GPU 上 prepare 或 benchmark；实际主实验需先生成 4 个 manual checkpoint。

## 2026-06-02 - Main 实验命令整理
- 开发目的：将 Qwen3.5-2B 五个主实验场景的 prepare/benchmark 命令和手动 hybrid 方案集中记录到 `artifacts/results/main`。
- 修改内容：新增主实验说明文档，所有 checkpoint 与 speed CSV 输出路径均指向 `artifacts/results/main/qwen3_5_2b/...`，环境命令改为本机 `conda activate cospaq`。
- 影响文件：`artifacts/results/main/qwen3_5_2b_manual_hybrid_experiment.md`、`dev/impls/034_qwen35_shape_workload_hybrid_impl.md`。
- 后续注意：文档只提供命令，没有实际运行 benchmark。

## 2026-06-02 - Main 结果分析
- 开发目的：汇总已完成的 Qwen3.5-2B 五场景主实验结果，判断当前 manual hybrid 是否优于单一方法。
- 修改内容：新增结果分析文档，按场景列出端到端、prefill、decode 指标，并给出当前 hybrid 慢于 dense 的结论与后续实验建议。
- 影响文件：`artifacts/results/main/qwen3_5_2b_result_analysis.md`、`dev/impls/034_qwen35_shape_workload_hybrid_impl.md`。
- 后续注意：当前每个方法每个场景只有一次测量，结论没有方差估计；若用于论文主结果，建议补重复运行或报告稳定性。

## 2026-06-02 - Kernel-driven hybrid 方案修正
- 开发目的：根据已完成的端到端 stage 结果和 Qwen3.5-2B 真实 linear shape，解释旧 manual hybrid 变慢的原因，并重新选择更有利的场景与 hybrid 策略。
- 修改内容：新增修正分析文档，指出旧方案错误地在 decode 使用 Marlin、在 prefill 过多使用 NVFP4；建议改为 prefill 使用 sparse BF16 但小 N projection 保持 BF16，decode 全部 BF16 dense，并选择 `B=16,input=1024,output=1` 的 prefill-dominant 场景。
- 影响文件：`artifacts/results/main/qwen3_5_2b_kernel_driven_revision.md`、`dev/impls/034_qwen35_shape_workload_hybrid_impl.md`。
- 后续注意：新建议还未实现成 `manual_hybrid_prefill_sparse_decode_dense` method，也未实际运行该场景。

## 2026-06-02 - 主命令文档补充 linear shape 策略
- 开发目的：在准备执行命令的主 md 文件中补充 Qwen3.5-2B 可压缩 linear 类型、shape 和不同 M 场景下的推荐压缩策略。
- 修改内容：新增 `Qwen3.5-2B Linear Shape 与场景策略` 章节，合并重复 layer，列出 12 类 linear 的 count/N/K，并区分 standalone kernel 表策略与端到端保守最强策略。
- 影响文件：`artifacts/results/main/qwen3_5_2b_manual_hybrid_experiment.md`、`dev/impls/034_qwen35_shape_workload_hybrid_impl.md`。
- 后续注意：章节中提出的 `F_prefill_dominant_short_decode` 和新 conservative hybrid 仍需实现 method 后实测。

## 2026-06-02 - 主命令文档细化典型 M 分析
- 开发目的：让主实验命令文档更清楚地说明如何从 kernel-level 结果推导场景选择和 hybrid 策略。
- 修改内容：扩展 `Qwen3.5-2B Linear Shape 与场景策略` 章节，增加 `M=1/4/8/16/2048/8192/16384` 的逐场景 backend 分组、代表性 speedup 范围，以及候选场景优先级。
- 影响文件：`artifacts/results/main/qwen3_5_2b_manual_hybrid_experiment.md`、`dev/impls/034_qwen35_shape_workload_hybrid_impl.md`。
- 后续注意：这些 speedup 是 standalone GEMM 层面的分析，仍需用端到端 benchmark 验证最终主实验场景。

## 2026-06-02 - 可压缩 Linear 表增加 M 策略列
- 开发目的：让主实验文档中的可压缩 linear 表直接展示不同典型 M 下每类 linear 的 kernel-level 最优 backend。
- 修改内容：在 `可压缩 Linear 类型` 表中新增 `M=1/4/8/16/2048/8192/16384` 列，并标注 exact benchmark 与 speed model 预测来源的区别。
- 影响文件：`artifacts/results/main/qwen3_5_2b_manual_hybrid_experiment.md`、`dev/impls/034_qwen35_shape_workload_hybrid_impl.md`。
- 后续注意：带 `*` 的格子不是 exact shape 直接实测，而是基于真实 kernel benchmark CSV 的 speed model 预测。

## 2026-06-02 - 可压缩 Linear 表补充次优与加速比
- 开发目的：让每个典型 M 下的 linear 策略矩阵同时展示最优、次优 backend 以及相对 BF16 的 kernel-level 加速比。
- 修改内容：更新 `可压缩 Linear 类型` 表，每个 M 单元格写入 top-2 backend，格式为 `backend (speedup)`；继续用 `*` 标注 speed model 预测来源。
- 影响文件：`artifacts/results/main/qwen3_5_2b_manual_hybrid_experiment.md`、`dev/impls/034_qwen35_shape_workload_hybrid_impl.md`。
- 后续注意：部分 shape/M 组合只有一个可用 backend，因此单元格只显示一项。

## 2026-06-02 - B1 I8192 O32 单场景策略分析
- 开发目的：基于可压缩 linear 表，针对 `batch=1,input=8192,output=32` 分析每类 linear 的 prefill/decode 推荐 backend。
- 修改内容：在主实验文档中新增单场景分析，分别给出理论最快两套表示方案、dense NVFP4 W4A4/W4A16 兼容切换推荐方案、以及完全不切换的单 backend per-linear 方案，并补充 kernel-level linear 总量估计。
- 影响文件：`artifacts/results/main/qwen3_5_2b_manual_hybrid_experiment.md`、`dev/impls/034_qwen35_shape_workload_hybrid_impl.md`。
- 后续注意：该分析仍是 kernel-level linear 估计；端到端性能需要单独 benchmark 验证。

## 2026-06-02 - 推荐 Hybrid Method 与单场景测试命令
- 开发目的：为 `batch=1,input=8192,output=32` 场景实现可测试的推荐 hybrid，并整理与单一压缩方法对比的真实 benchmark 命令。
- 修改内容：新增 `hybrid_nvfp4_major` method，只对主要大 linear 使用 dense NVFP4 W4A4/W4A16 动态切换，保留 `linear_attn.in_proj_a/b` 和 `self_attn.k/v_proj` 为 BF16；主实验文档中新增该场景的 prepare/benchmark 命令。
- 影响文件：`fake/models/qwen3_5_kernels.py`、`artifacts/results/main/qwen3_5_2b_manual_hybrid_experiment.md`、`dev/impls/034_qwen35_shape_workload_hybrid_impl.md`。
- 后续注意：已通过 `py_compile` 和 prepare/bench CLI help 检查；尚未实际 prepare checkpoint 或运行 benchmark。

## 2026-06-02 - G_b1_i8192_o32 结果分析
- 开发目的：分析已完成的 `batch=1,input=8192,output=32` 单场景 benchmark，判断推荐 `hybrid_nvfp4_major` 是否优于单一方法。
- 修改内容：在结果分析文档中新增该场景结果表和解释，指出 dense 仍为端到端最快，`hybrid_nvfp4_major` 只优于 `dense_nvfp4/sparse_bf16/sparse_nvfp4`，不优于 dense 或 full Marlin。
- 影响文件：`artifacts/results/main/qwen3_5_2b_result_analysis.md`、`dev/impls/034_qwen35_shape_workload_hybrid_impl.md`。
- 后续注意：该结果说明 Qwen3.5-2B 的 kernel-level NVFP4 预期与端到端实际不一致，不能作为正向主实验结论。

## 2026-06-02 - Kernel-level 与端到端 mismatch 原因分析
- 开发目的：解释为什么 `5_kernel_comprehensive` 的 kernel-level 预测没有在 Qwen3.5-2B 真实 prefill/decode 中兑现。
- 修改内容：在结果分析文档中补充 mismatch 原因，指出 kernel benchmark 是 GEMM-only，dense/sparse NVFP4 的 activation packing 在计时外，而真实 module forward 每次都包含 packing；同时 decode 端到端中可替换 linear 占比低。
- 影响文件：`artifacts/results/main/qwen3_5_2b_result_analysis.md`、`dev/impls/034_qwen35_shape_workload_hybrid_impl.md`。
- 后续注意：后续应补 module-level forward benchmark，直接测封装后的 `Linear.forward`，再据此做 hybrid 路由。

## 2026-06-02 - Module-level 5 Kernel Benchmark 脚本
- 开发目的：测试 5 种 kernel 封装成真实 Linear module 后的 forward 速度，补齐 GEMM-only benchmark 未包含 activation packing/wrapper 开销的问题。
- 修改内容：新增 `bench_5_kernel_modules_comprehensive.py`，按 fixed `m/n/k` 三维扫描 shape，计时 `module(x)`；新增 `run_5_kernel_modules_3gpu.sh`，把 fixed `m/n/k` 分别放到 GPU 1/2/3 并行运行；主实验文档补充运行命令和输出文件说明。
- 影响文件：`fake/kernels/cutlass/cutlass_wrapper/benchmarks/bench_5_kernel_modules_comprehensive.py`、`fake/kernels/cutlass/cutlass_wrapper/benchmarks/run_5_kernel_modules_3gpu.sh`、`artifacts/results/main/qwen3_5_2b_manual_hybrid_experiment.md`、`dev/impls/034_qwen35_shape_workload_hybrid_impl.md`。
- 后续注意：当前机器 CUDA 不可用，仅完成 `py_compile`、脚本 `--help`、bash 语法检查；实际 benchmark 需在 GPU 1/2/3 可用机器上运行。

## 2026-06-02 - Module-level Kernel Benchmark 可视化
- 开发目的：将已完成的 module-forward 5 kernel benchmark 结果按旧 `5_kernel_comprehensive` 风格可视化。
- 修改内容：新增 `visualize_module_kernel_benchmarks.py`，生成 fixed `m/n/k` 三张 fastest-kernel heatmap、三张 best speedup-vs-BF16 heatmap，以及一张 module roofline 图。
- 影响文件：`artifacts/results/benchmarks/kernel/visualize_module_kernel_benchmarks.py`、`artifacts/results/benchmarks/kernel/module_heatmap_fix_m.png`、`artifacts/results/benchmarks/kernel/module_heatmap_fix_n.png`、`artifacts/results/benchmarks/kernel/module_heatmap_fix_k.png`、`artifacts/results/benchmarks/kernel/module_speedup_fix_m.png`、`artifacts/results/benchmarks/kernel/module_speedup_fix_n.png`、`artifacts/results/benchmarks/kernel/module_speedup_fix_k.png`、`artifacts/results/benchmarks/kernel/module_roofline.png`。
- 后续注意：`dense_nvfp4` 在 fixed `m/n` 中存在少量 error 行，后续分析时应检查对应 log/error_msg，避免误读为 skip。

## 2026-06-02 - Qwen3.5-2B Module Kernel M Sweep
- 开发目的：针对 Qwen3.5-2B 可压缩 linear 的真实权重 shape，测试封装后 `Linear.forward` 随 token M 增大的 5 kernel 延迟曲线。
- 修改内容：新增 `bench_qwen3_5_2b_module_kernels.py`，固定 Qwen3.5-2B 的 12 类可压缩 linear shape，扫描 `M=1..16384`，计时真实 module `forward`；sparse BF16/NVFP4 使用 Qwen runtime 的 padded wrapper，CSV 仍按标准 `M=tokens,N=out_features,K=in_features` 记录。新增结果目录 README，记录运行命令和 shape 语义。
- 影响文件：`scripts/bench_qwen3_5_2b_module_kernels.py`、`artifacts/results/benchmarks/module/Qwen3.5-2B/kernel/README.md`、`dev/impls/034_qwen35_shape_workload_hybrid_impl.md`。
- 后续注意：当前机器未执行 GPU benchmark；已完成 `py_compile` 和 `--help` 检查，实际结果需在 CUDA 可用机器运行后生成 CSV。

## 2026-06-02 - Qwen3.5-2B Module Kernel 曲线可视化
- 开发目的：将 Qwen3.5-2B 各可压缩 linear shape 下 5 个 packaged kernel 随 M 增长的 forward 延迟可视化为曲线图。
- 修改内容：新增 `visualize_qwen35_2b_module_kernel_curves.py`，生成 3x4 子图布局，每个子图对应一个 linear shape，子图内绘制 5 个 kernel 的 latency-vs-M 曲线；输出 PNG 和 PDF。
- 影响文件：`artifacts/results/benchmarks/module/Qwen3.5-2B/kernel/visualize_qwen35_2b_module_kernel_curves.py`、`artifacts/results/benchmarks/module/Qwen3.5-2B/kernel/qwen35_2b_module_kernel_latency_curves.png`、`artifacts/results/benchmarks/module/Qwen3.5-2B/kernel/qwen35_2b_module_kernel_latency_curves.pdf`、`dev/impls/034_qwen35_shape_workload_hybrid_impl.md`。
- 后续注意：不支持或运行失败的 kernel 不绘制曲线；图例固定显示 5 个 kernel，便于跨子图比较。

## 2026-06-02 - Qwen3.5 4B/9B/27B Module Kernel Benchmark 扩展
- 开发目的：将 Qwen3.5-2B 的 module-forward kernel M sweep 扩展到 Qwen3.5-4B、9B、27B，并分别生成同形式曲线图。
- 修改内容：把 `bench_qwen3_5_2b_module_kernels.py` 泛化为可指定 `--model-name/--model-path`，从 safetensors 真实权重 shape 中提取 `model.language_model.layers.*` 的可压缩 linear，排除 `mtp.layers.*`；新增通用入口 `bench_qwen3_5_module_kernels.py`、通用绘图脚本 `visualize_qwen3_5_module_kernel_curves.py`、4B/9B/27B 并行运行脚本和说明文档。
- 影响文件：`scripts/bench_qwen3_5_2b_module_kernels.py`、`scripts/bench_qwen3_5_module_kernels.py`、`scripts/visualize_qwen3_5_module_kernel_curves.py`、`scripts/run_qwen3_5_module_kernel_benchmarks_4b_9b_27b.sh`、`artifacts/results/benchmarks/module/Qwen3.5_4B_9B_27B_module_kernel_benchmarks.md`、`dev/impls/034_qwen35_shape_workload_hybrid_impl.md`。
- 后续注意：当前环境 CUDA 不可用，没有实际启动 4B/9B/27B benchmark；已完成 Python 编译、bash 语法检查和 `--print-shapes` shape 提取验证。实际运行后会在各模型 `artifacts/results/benchmarks/module/Qwen3.5-{4B,9B,27B}/kernel/` 下生成 CSV/PNG/PDF。

## 2026-06-03 - Qwen3.5-4B B1024 I32 O32 真实推理补测
- 开发目的：补测 `batch_size=1024,input_tokens=32,output_tokens=32` 下 Qwen3.5-4B 五种真实 kernel/runtime 的端到端推理速度。
- 修改内容：在本机 `cospaq` 环境和 RTX 5090 上顺序运行 `dense`、`dense_nvfp4`、`sparse_bf16`、`sparse_nvfp4`、`marlin_nvfp4` 单场景 benchmark；五种方法均因显存不足写入 `status=OOM`，并追加到各自 `speed.csv`。
- 影响文件：`artifacts/results/qwen3_5_4b_dense/speed.csv`、`artifacts/results/qwen3_5_4b_dense_nvfp4/speed.csv`、`artifacts/results/qwen3_5_4b_sparse_bf16/speed.csv`、`artifacts/results/qwen3_5_4b_sparse_nvfp4/speed.csv`、`artifacts/results/qwen3_5_4b_marlin_nvfp4/speed.csv`、`dev/impls/034_qwen35_shape_workload_hybrid_impl.md`。
- 后续注意：本场景在单张 32GB RTX 5090 上无法完成；如必须得到非 OOM 性能，需要降低 batch、启用多 GPU/device_map，或改测 module-level `M=32768`/`M=1024` 近似分析。

## 2026-06-03 - Qwen3.5-4B Module Kernel 重新测试
- 开发目的：在当前 kernel 修改后，重新测试 Qwen3.5-4B 所有可压缩 Linear shape 的 packaged `Linear.forward` 速度。
- 修改内容：使用 `cospaq` 环境和 RTX 5090 运行 `scripts/bench_qwen3_5_module_kernels.py --model-name Qwen3.5-4B --warmup 5 --iters 20`，覆盖生成 12 类 Linear、15 个 M 值、5 个 kernel 的 CSV，并重新生成 PNG/PDF 曲线图。
- 影响文件：`artifacts/results/benchmarks/module/Qwen3.5-4B/kernel/qwen35_4b_module_kernel_curves.csv`、`artifacts/results/benchmarks/module/Qwen3.5-4B/kernel/qwen35_4b_module_kernel_latency_curves.png`、`artifacts/results/benchmarks/module/Qwen3.5-4B/kernel/qwen35_4b_module_kernel_latency_curves.pdf`、`dev/impls/034_qwen35_shape_workload_hybrid_impl.md`。
- 后续注意：本次结果共 900 行，870 个 `pass`、30 个 `skip`，skip 全部来自 `marlin_nvfp4` 不支持 `N=32,K=2560` 的 `linear_attn.in_proj_a/b`；没有 `error` 行。

## 2026-06-03 - Qwen3.5-9B Module Kernel 重新测试
- 开发目的：在当前 kernel 修改后，重新测试 Qwen3.5-9B 所有可压缩 Linear shape 的 packaged `Linear.forward` 速度。
- 修改内容：使用 `cospaq` 环境和 RTX 5090 的物理 GPU 5/6/7，将 M sweep 分为 `1 2 4 8 16`、`32 64 128 256 512`、`1024 2048 4096 8192 16384` 三段并行运行，再合并为标准 CSV，并重新生成 PNG/PDF 曲线图。
- 影响文件：`artifacts/results/benchmarks/module/Qwen3.5-9B/kernel/qwen35_9b_module_kernel_curves.csv`、`artifacts/results/benchmarks/module/Qwen3.5-9B/kernel/qwen35_9b_module_kernel_latency_curves.png`、`artifacts/results/benchmarks/module/Qwen3.5-9B/kernel/qwen35_9b_module_kernel_latency_curves.pdf`、`artifacts/results/benchmarks/module/Qwen3.5-9B/kernel/tmp_shards/`、`dev/impls/034_qwen35_shape_workload_hybrid_impl.md`。
- 后续注意：本次结果共 900 行，855 个 `pass`、30 个 `skip`、15 个 `error`；skip 全部来自 `marlin_nvfp4` 不支持 `N=32,K=4096` 的 `linear_attn.in_proj_a/b`；error 来自 `sparse_bf16` 在 `self_attn.k_proj/v_proj` 的 `N=1024,K=4096` shape 上触发 cuSPARSELt 2:4 prune check 失败。
