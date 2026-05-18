## 2026-05-16 - CUTLASS sparse storage checkpoint layer
- 开发目的：将省磁盘保存格式和 CUTLASS kernel-ready runtime 格式分层，解释并改善 sparse runtime checkpoint 比 dense NVFP4 更大的问题。
- 修改内容：新增 sparse storage checkpoint compact pairwise 4:8 FP4 表示、storage 导出脚本、storage-to-runtime 转换入口和 Slurm/README 说明。
- 影响文件：`fake/models/dinov3_cutlass_storage.py`、`scripts/prepare_dinov3_cutlass_storage_checkpoint.py`、`scripts/prepare_dinov3_cutlass_runtime_checkpoint.py`、相关 Slurm 和文档。
- 后续注意：storage checkpoint 可直接作为加载入口；也可显式转换成 runtime checkpoint 做缓存/调试。

## 2026-05-16 - Storage load-time packing path
- 开发目的：让 sparse storage checkpoint 成为真实推理入口，加载模型时在内存中转换为 CUTLASS runtime buffers，避免必须额外保存 runtime checkpoint。
- 修改内容：新增 `load_dinov3_vit7b16_cutlass_storage_classifier()`；sparse speed/accuracy 支持 `--storage-checkpoint`，Slurm 支持 `STORAGE_CHECKPOINT`；修复 meta skeleton 加载后 non-persistent RoPE buffer 仍为 meta 导致 `.to(cuda)` 失败的问题。
- 影响文件：`fake/models/dinov3_cutlass_storage.py`、`fake/models/dinov3_cutlass_runtime.py`、sparse speed/accuracy 脚本、对应 Slurm、`scripts/README.md`。
- 后续注意：`STORAGE_CHECKPOINT` 与 `RUNTIME_CHECKPOINT` 互斥；CSV 中 `checkpoint_format=cutlass_storage_packed_v1` 且 `runtime_checkpoint_loader_mode=storage_loadtime_pack` 时表示未落盘 runtime checkpoint。

## 2026-05-16 - DINOv3 batch-size sweep entry
- 开发目的：批量测试 dense、CUTLASS dense NVFP4、CUTLASS sparse NVFP4 在不同 batch size 下的端到端 forward speed，寻找吞吐极限点。
- 修改内容：新增统一 Slurm 入口 `scripts/slurm/bench_dinov3_vit7b16_batch_sweep_speed.sh`，支持 `METHODS`、`BATCH_SIZES`、`WARMUP`、`ITERS`、`STOP_ON_FAIL` 和各路径输出 CSV 覆盖。
- 影响文件：`scripts/slurm/bench_dinov3_vit7b16_batch_sweep_speed.sh`、`dev/impls/014_cutlass_storage_checkpoint_impl.md`。
- 后续注意：默认结果 append 到各自 speed CSV；大 batch OOM 时默认停止当前方法后续更大 batch。
