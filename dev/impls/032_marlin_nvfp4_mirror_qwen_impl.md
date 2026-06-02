## 2026-06-01 - Marlin NVFP4 MIRROR/Qwen 接入
- 开发目的：将新增 W4A16 Marlin NVFP4 weight-only kernel 接入主框架，并覆盖 MIRROR 与 Qwen3.5 端到端速度路径。
- 修改内容：新增 Marlin NVFP4 packed checkpoint 构建/加载；扩展 Qwen3.5 language model Linear 选择；MIRROR speed/accuracy 增加 `marlin_nvfp4`；Qwen3.5 benchmark 增加 `--method dense|marlin_nvfp4` 和 `--variant`。
- 影响文件：`fake/kernels/marlin_nvfp4.py`、`fake/compression/modules.py`、`fake/models/qwen3_5.py`、`scripts/prepare_marlin_nvfp4_checkpoint.py`、`scripts/bench_mirror_compressed_speed.py`、`scripts/eval_mirror_compressed_accuracy.py`、`scripts/bench_qwen3_5_speed.py`、`scripts/slurm/prepare_mirror_compressed_models.sh`、`scripts/slurm/bench_mirror_compressed_speed.sh`、`scripts/slurm/bench_mirror_batch_sweep_speed.sh`、`scripts/slurm/eval_mirror_compressed_accuracy.sh`。
- 后续注意：`marlin_nvfp4` runtime 现在要求预先运行 prepare 脚本生成 packed checkpoint，不再从 dense BF16 权重在线 pack；当前执行环境 `torch.cuda.is_available()` 为 False，GPU smoke 未能完成。
