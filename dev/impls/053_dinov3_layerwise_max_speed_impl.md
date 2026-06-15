## 2026-06-15 - DINOv3 layerwise max-speed scaffolding
- 开发目的：实现基于速度模型的 DINOv3 分层最优 CUTLASS 后端选择，并能在超算 GPU 节点实测整模型 forward 速度。
- 修改内容：在 `artifacts/debug/019_dinov3_layerwise_max_speed/code/` 下新增 DINO layerwise policy 替换模块、策略生成/benchmark/summary runner、Slurm 脚本和 053 plan 文件。
- 影响文件：`artifacts/debug/019_dinov3_layerwise_max_speed/code/dinov3_layerwise_policy.py`、`artifacts/debug/019_dinov3_layerwise_max_speed/code/run_dinov3_layerwise_max_speed.py`、`artifacts/debug/019_dinov3_layerwise_max_speed/code/run_dinov3_layerwise_max_speed.sh`、`dev/plans/053_dinov3_layerwise_max_speed_plan.md`。
- 后续注意：本机缺少 DINO 权重时只能做静态验证；完整速度需要在 RTX 5090 超算节点运行 Slurm 脚本。

## 2026-06-15 - 超算路径导入修复
- 开发目的：修复超算运行时报 `ModuleNotFoundError: No module named 'modeling'`，并避免 Slurm 脚本硬编码仓库路径。
- 修改内容：runner 从脚本位置向上发现仓库根目录，并自动搜索 `fake/**/modeling/kernel_predictor.py` 后加入 import path；Slurm 脚本改为从自身位置反推 `REPO_ROOT` 后进入仓库。
- 影响文件：`artifacts/debug/019_dinov3_layerwise_max_speed/code/run_dinov3_layerwise_max_speed.py`、`artifacts/debug/019_dinov3_layerwise_max_speed/code/run_dinov3_layerwise_max_speed.sh`。
- 后续注意：如果超算仓库确实缺少 `fake/kernels/cutlass/cutlass_wrapper/modeling/kernel_predictor.py`，新报错会打印搜索过的路径。

## 2026-06-15 - 旧手动 hybrid 对比柱状图
- 开发目的：先可视化已有 DINOv3 手动 hybrid 速度结果，对比单一 CUTLASS 方法和手动混合方法。
- 修改内容：新增绘图脚本，读取已有 dense、dense NVFP4、sparse BF16、sparse NVFP4 和 manual hybrid CSV，在 batch 16/32 上输出吞吐与延迟柱状图、汇总 CSV 和 README。
- 影响文件：`artifacts/debug/019_dinov3_layerwise_max_speed/code/plot_previous_dinov3_manual_hybrid_bars.py`、`artifacts/debug/019_dinov3_layerwise_max_speed/previous_manual_hybrid/`。
- 后续注意：manual hybrid 旧结果的 `iters=10`，单一方法多为 `iters=20`；图中保留该元数据，主要用于快速趋势对比。

## 2026-06-15 - hybrid 图展示调整
- 开发目的：按展示要求去掉图中文字中的 manual/previous，并避免 batch 16 缺 sparse BF16 导致对比不完整。
- 修改内容：绘图脚本只绘制 batch 32；图例中的混合方法命名为 `Hybrid`；输出目录改为 `hybrid_vs_uniform`，图片和汇总文件名去掉 manual/previous；汇总 CSV 中的内部展示 key 也改为 `hybrid` / `b32`。
- 影响文件：`artifacts/debug/019_dinov3_layerwise_max_speed/code/plot_dinov3_hybrid_bars.py`、`artifacts/debug/019_dinov3_layerwise_max_speed/hybrid_vs_uniform/`。
- 后续注意：旧的 `previous_manual_hybrid/` 目录如果已生成，可忽略或手动删除；新图以后者为准。

## 2026-06-15 - hybrid 图美化和说明补充
- 开发目的：优化 batch 32 对比柱状图的展示，并解释 DINOv3 hybrid 收益小于 LLaMA prefill-only 的原因。
- 修改内容：图例改为图外单行展示，柱顶只保留数值并去掉 `b32` 标注；README 增加 DINOv3 与 LLaMA-2 prefill-only 收益差异说明。
- 影响文件：`artifacts/debug/019_dinov3_layerwise_max_speed/code/plot_dinov3_hybrid_bars.py`、`artifacts/debug/019_dinov3_layerwise_max_speed/hybrid_vs_uniform/README.md`、两张 `hybrid_vs_uniform_*.png`。
- 后续注意：DINOv3 旧 hybrid 结果的 `iters=10`，单一方法多为 `iters=20`，表中仍保留元数据。

## 2026-06-15 - Slurm 写权限修复和 Hybrid 精度入口
- 开发目的：修复超算运行时在当前目录创建 `out/err/artifacts` 权限不足的问题，并将本轮速度测试收敛到 batch 32，同时补测已有 Hybrid 的 ImageNet 精度。
- 修改内容：Slurm 脚本默认 `BATCH_SIZES=32`；`out/err` 创建失败不再中断；`OUTPUT_ROOT` 自动选择可写目录，优先仓库 `artifacts/debug/019...`，其次 `SLURM_SUBMIT_DIR`，最后 `$HOME/dinov3_layerwise_max_speed_019`；新增 `eval_dinov3_hybrid_accuracy.py`，默认评估 `b32_manual` Hybrid 并输出 `hybrid_accuracy.csv`。
- 影响文件：`artifacts/debug/019_dinov3_layerwise_max_speed/code/run_dinov3_layerwise_max_speed.sh`、`artifacts/debug/019_dinov3_layerwise_max_speed/code/eval_dinov3_hybrid_accuracy.py`。
- 后续注意：如果希望跳过精度只跑速度，提交时设置 `RUN_ACCURACY=0`；如果希望强制输出到指定位置，设置 `OUTPUT_ROOT=/path/to/writable/dir`。
