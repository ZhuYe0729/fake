## 2026-05-14 - DINOv3 ViT-7B NVFP4 microbenchmark

- 开发目的：为 DINOv3 ViT-7B/16 增加 FlashInfer NVFP4 逐层 microbenchmark，与 MaxViT 的分析输出保持相近格式。
- 修改内容：新增 `fake/models/dinov3_nvfp4.py` runtime loader，新增 `scripts/bench_dinov3_vit7b16_nvfp4_micro.py`，新增 `scripts/slurm/analysis/bench_dinov3_vit7b16_nvfp4_micro.sh`，并补充 `scripts/README.md` 的 DINOv3/NVFP4 条目。
- 影响文件：`fake/models/dinov3_nvfp4.py`、`scripts/bench_dinov3_vit7b16_nvfp4_micro.py`、`scripts/slurm/analysis/bench_dinov3_vit7b16_nvfp4_micro.sh`、`scripts/README.md`、`dev/plans/008_dinov3_nvfp4_microbenchmark_plan.md`。
- 后续注意：DINOv3 7B 全量逐层 microbenchmark 运行时间较长，正式跑前建议先用 `MAX_LAYERS=3 WARMUP=2 ITERS=5` 冒烟。

## 2026-05-14 - 修复 DINOv3 NVFP4 权重 dtype

- 开发目的：修复 DINOv3 ViT-7B NVFP4 替换时报 `fp4_quantize only supports input tensor with dtypes fp16/bf16/e4m3` 的问题。
- 修改内容：DINOv3 NVFP4 loader 增加 `dtype` 参数并默认转为 bf16 后再替换 Linear；benchmark 脚本和 Slurm 脚本增加 `--dtype`/`DTYPE` 参数；README 标明默认 bf16。
- 影响文件：`fake/models/dinov3_nvfp4.py`、`scripts/bench_dinov3_vit7b16_nvfp4_micro.py`、`scripts/slurm/analysis/bench_dinov3_vit7b16_nvfp4_micro.sh`、`scripts/README.md`。
- 后续注意：FlashInfer NVFP4 quantize 不支持 fp32 输入，DINOv3 相关 NVFP4 路径必须保持 bf16 或 fp16。

## 2026-05-14 - 支持多 batch size 与错误记录

- 开发目的：让 DINOv3 ViT-7B NVFP4 microbenchmark 可一次测试多个 batch size，并在 OOM 等单个配置失败时写入 CSV 标记。
- 修改内容：`bench_dinov3_vit7b16_nvfp4_micro.py` 新增 `--batch-sizes`；每个 batch/input size 独立执行，成功行写 `status=OK`，失败行写 `status=ERROR`、`error_type`、`error_message`；Slurm analysis 脚本新增 `BATCH_SIZES` 并传给 Python。
- 影响文件：`scripts/bench_dinov3_vit7b16_nvfp4_micro.py`、`scripts/slurm/analysis/bench_dinov3_vit7b16_nvfp4_micro.sh`、`scripts/README.md`。
- 后续注意：7B 模型在大 batch 下容易 OOM，建议先用较少输入尺寸或较小 `ITERS` 跑完整 batch sweep。

## 2026-05-14 - DINOv3 microbenchmark 按 shape 去重

- 开发目的：避免 DINOv3 7B 中大量结构相同的 Linear 层重复做 microbenchmark，缩短测试时间。
- 修改内容：捕获所有层调用后，按 `(input_shape, m, n, k)` 去重，只 benchmark 每种唯一 GEMM shape 的第一次层调用；日志输出增加 `unique_layer_calls` 与 `raw_layer_calls`。
- 影响文件：`scripts/bench_dinov3_vit7b16_nvfp4_micro.py`。
- 后续注意：CSV 不再逐层覆盖所有同构层，结果代表每种 shape 的一次样本测量。
