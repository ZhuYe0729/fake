# 032 Marlin NVFP4 MIRROR And Qwen3.5 Integration Plan

## Summary
把新增 W4A16 Marlin NVFP4 weight-only kernel 接入主框架，范围为通用 Linear replacement、Marlin packed checkpoint 准备、MIRROR 鉴伪端到端、Qwen3.5 文本 prefill/decode 端到端速度。新增方法名统一为 `marlin_nvfp4`。

## Key Changes
- 新增 `fake/kernels/marlin_nvfp4.py`，提供 `MarlinNVFP4Config`、replacement report、可用性检查、模块计数、packed checkpoint 构建和 packed checkpoint 加载。
- 新增 `scripts/prepare_marlin_nvfp4_checkpoint.py`，在准备/压缩阶段把 dense 权重 pack 成 Marlin NVFP4 checkpoint。
- 扩展 `select_compressible_modules()` 支持 `qwen3_5`，只选择 language model 内部 Linear，排除 vision、embedding 和 `lm_head`。
- 在 MIRROR speed/accuracy 脚本中加入 `marlin_nvfp4`，runtime 只加载 `artifacts/checkpoints/mirror/marlin_nvfp4/model.pt`，不在线 pack。
- 在 Qwen3.5 benchmark 中加入 `--method dense|marlin_nvfp4` 和 `--variant`，默认路径为 `/home/agent/wja/data/models/Qwen/Qwen3.5-0.8B`，支持 0.8B/2B/4B/9B/27B。
- 更新本机 GPU 运行口径，默认使用 `conda activate cospaq` 和直接 Python 命令；Slurm 脚本只做兼容保留。

## Test Plan
- `conda activate cospaq`
- `PYTHONPATH=fake/kernels/cutlass/cutlass_wrapper pytest fake/kernels/cutlass/cutlass_wrapper/tests/test_marlin_nvfp4_linear.py -q`
- `PYTHONPATH=. python scripts/prepare_marlin_nvfp4_checkpoint.py --model mirror --dtype bf16`
- `PYTHONPATH=. python scripts/prepare_marlin_nvfp4_checkpoint.py --model qwen3_5 --qwen-variant 0.8B --dtype bf16`
- `PYTHONPATH=. python scripts/bench_mirror_compressed_speed.py --method marlin_nvfp4 --batch-size 1 --warmup 2 --iters 5 --output artifacts/results/mirror_compressed/speed_marlin_smoke.csv`
- `PYTHONPATH=. python scripts/eval_mirror_compressed_accuracy.py --method marlin_nvfp4 --benchmarks Chameleon GenImage --limit-per-class 8 --batch-size 8 --num-workers 2 --output artifacts/results/mirror_compressed/accuracy_marlin_smoke.csv`
- `PYTHONPATH=. python scripts/bench_qwen3_5_speed.py --method marlin_nvfp4 --variant 0.8B --input-tokens 128 --output-tokens 16 --batch-sizes 1 --warmup 2 --iters 5`

## Assumptions
- Qwen 首版做文本-only speed；现有 multimodal benchmark 保留但 `marlin_nvfp4` 只替换 language model Linear。
- `marlin_nvfp4` 是 W4A16 weight-only，activation 保持 BF16/FP16。
- 首版 packed checkpoint 保存完整 model state_dict，其中目标 Linear 已替换为 Marlin packed buffers；runtime 不重新量化/pack。
- 不支持 shape 只记录 skipped，不中断整体评测。
