# Scripts

这个目录放项目级的可执行脚本。后续新增脚本时，在这里补一条简短说明即可；复杂实验细节放到对应的 `dev/plans/` 和 `dev/impls/` 记录里。

大多数 benchmark/eval 脚本需要在 GPU 计算节点运行。登录节点通常只适合做语法检查、准备文件和提交 Slurm 作业。

## MaxViT / NVFP4

- `bench_maxvit_dense_speed.py`：MaxViT dense 端到端 forward benchmark。
- `bench_maxvit_nvfp4_speed.py`：MaxViT FlashInfer NVFP4 端到端 forward benchmark。
- `bench_maxvit_cutlass_nvfp4_speed.py`：MaxViT CUTLASS dense NVFP4 端到端 forward benchmark。
- `eval_maxvit_cutlass_nvfp4_accuracy.py`：MaxViT CUTLASS dense NVFP4 ImageNet accuracy。
- `bench_maxvit_cutlass_sparse_nvfp4_speed.py`：MaxViT CUTLASS sparse NVFP4 端到端 forward benchmark。
- `eval_maxvit_cutlass_sparse_nvfp4_accuracy.py`：MaxViT CUTLASS sparse NVFP4 ImageNet accuracy。
- `bench_maxvit_cutlass_sparse_bf16_speed.py`：MaxViT CUTLASS/cuSPARSELt sparse BF16 端到端 forward benchmark。
- `eval_maxvit_cutlass_sparse_bf16_accuracy.py`：MaxViT CUTLASS/cuSPARSELt sparse BF16 ImageNet accuracy。
- `bench_maxvit_nvfp4_micro.py`：逐层拆解 MaxViT NVFP4 Linear 的耗时，适合定位 quant、GEMM、scale、bias 等组件开销。
- `compare_maxvit_nvfp4_outputs.py`：随机输入下比较 dense 与 NVFP4 logits 差异。
- `check_flashinfer_nvfp4.py`：最小矩阵上的 FlashInfer `nvfp4_quantize` 与 `mm_fp4` smoke test。

`bench_maxvit_nvfp4_micro.py` 默认测试 MaxViT tiny，在 `3x224x224`、`3x448x448`、`3x672x672` 三种输入尺寸下捕获所有 NVFP4 Linear 的真实输入 shape，并把 `layer_forward`、`activation_quant_only`、`gemm_only`、`dense_linear` 等 op 写入 CSV。脚本会按模型实际 window/partition size 校验输入；MaxViT large 默认合法输入为 `3x512x512`。可用 `--batch-sizes` 或 Slurm 环境变量 `BATCH_SIZES` 一次测试多个 batch size；单个配置失败时会写入 `status=ERROR`。

默认输出：

```text
artifacts/results/maxvit_tiny_nvfp4/microbench.csv
```

快速冒烟：

```bash
python scripts/bench_maxvit_nvfp4_micro.py --max-layers 3 --warmup 5 --iters 10
```

MaxViT CUTLASS 路径独立于现有 FlashInfer NVFP4 路径。首版只替换 MaxViT 中的 Linear；`select_compressible_modules`
里识别到的 MBConv pointwise Conv2d 仍保持 dense，因此 CUTLASS CSV 的 skipped count 可能包含 `unsupported_kind:conv2d`。
如果要使用磁盘上的真实压缩 checkpoint，先运行 `prepare_maxvit_cutlass_checkpoints.sh`。dense NVFP4 输出
`cutlass_runtime_packed_v1`，sparse NVFP4 输出 `cutlass_storage_packed_v1`，sparse BF16 输出
`cutlass_runtime_packed_v1`。

Slurm 冒烟：

```bash
MAXVIT_VARIANT=tiny BATCH_SIZE=8 WARMUP=1 ITERS=2 \
sbatch scripts/slurm/bench_maxvit_dense_speed.sh

MAXVIT_VARIANT=tiny BATCH_SIZE=8 WARMUP=1 ITERS=2 \
sbatch scripts/slurm/bench_maxvit_cutlass_nvfp4_speed.sh

MAXVIT_VARIANT=tiny BATCH_SIZE=8 WARMUP=1 ITERS=2 \
sbatch scripts/slurm/bench_maxvit_cutlass_sparse_nvfp4_speed.sh

MAXVIT_VARIANT=tiny BATCH_SIZE=8 WARMUP=1 ITERS=2 \
sbatch scripts/slurm/bench_maxvit_cutlass_sparse_bf16_speed.sh
```

真实压缩 checkpoint 默认输出：

```text
artifacts/checkpoints/maxvit_tiny/cutlass_nvfp4_runtime/model.pt
artifacts/checkpoints/maxvit_tiny/cutlass_sparse_nvfp4_storage/model.pt
artifacts/checkpoints/maxvit_tiny/cutlass_sparse_bf16_runtime/model.pt
```

默认输出：

```text
artifacts/results/maxvit_tiny_dense/speed.csv
artifacts/results/maxvit_tiny_dense/accuracy.csv
artifacts/results/maxvit_tiny_cutlass_nvfp4/speed.csv
artifacts/results/maxvit_tiny_cutlass_nvfp4/accuracy.csv
artifacts/results/maxvit_tiny_cutlass_sparse_nvfp4/speed.csv
artifacts/results/maxvit_tiny_cutlass_sparse_nvfp4/accuracy.csv
artifacts/results/maxvit_tiny_cutlass_sparse_bf16/speed.csv
artifacts/results/maxvit_tiny_cutlass_sparse_bf16/accuracy.csv
```

## DINOv3 / NVFP4

- `bench_dinov3_vit7b16_dense_speed.py`：DINOv3 ViT-7B/16 dense classifier forward benchmark。
- `bench_dinov3_vit7b16_nvfp4_micro.py`：逐层拆解 DINOv3 ViT-7B/16 backbone NVFP4 Linear 的耗时。
- `bench_dinov3_vit7b16_cutlass_nvfp4_speed.py`：DINOv3 ViT-7B/16 CUTLASS dense NVFP4 classifier forward benchmark，使用真实 CUTLASS NVFP4 kernel。
- `eval_dinov3_vit7b16_cutlass_nvfp4_accuracy.py`：DINOv3 ViT-7B/16 CUTLASS dense NVFP4 ImageNet linear-head accuracy。
- `bench_dinov3_vit7b16_cutlass_sparse_nvfp4_speed.py`：DINOv3 ViT-7B/16 CUTLASS structured sparse NVFP4 classifier forward benchmark。
- `eval_dinov3_vit7b16_cutlass_sparse_nvfp4_accuracy.py`：DINOv3 ViT-7B/16 CUTLASS structured sparse NVFP4 ImageNet linear-head accuracy。
- `bench_dinov3_vit7b16_cutlass_sparse_bf16_speed.py`：DINOv3 ViT-7B/16 CUTLASS/cuSPARSELt structured sparse BF16 classifier forward benchmark。
- `eval_dinov3_vit7b16_cutlass_sparse_bf16_accuracy.py`：DINOv3 ViT-7B/16 CUTLASS/cuSPARSELt structured sparse BF16 ImageNet linear-head accuracy。

`bench_dinov3_vit7b16_nvfp4_micro.py` 默认以 bf16 运行，输入尺寸为 `3x128x128`、`3x256x256`、`3x384x384`，支持 `--batch-sizes` / `BATCH_SIZES`，单个配置失败时会写入 `status=ERROR`。默认输出：

```text
artifacts/analysis/dinov3_vit7b16/nvfp4/microbench.csv
```

快速冒烟：

```bash
python scripts/bench_dinov3_vit7b16_nvfp4_micro.py --max-layers 3 --warmup 2 --iters 5
```

CUTLASS NVFP4 路径是独立真实 kernel 推理路径，不替换现有 FlashInfer 封装。它会把 DINOv3 backbone transformer 内 280 个 projection Linear 替换为 `fake/kernels/cutlass/cutlass_wrapper` 的 `NVFP4Linear`，classifier head 保持 bf16 dense。默认输出：

```text
artifacts/results/dinov3_vit7b16_cutlass_nvfp4/speed.csv
artifacts/results/dinov3_vit7b16_cutlass_nvfp4/accuracy.csv
```

Slurm 冒烟：

```bash
WARMUP=1 ITERS=2 sbatch scripts/slurm/bench_dinov3_vit7b16_cutlass_nvfp4_speed.sh
```

CUTLASS sparse NVFP4 路径同样独立于 FlashInfer 和 dense CUTLASS 路径，默认在转换时按 pairwise 4:8 magnitude pruning 生成结构化稀疏权重。DINOv3 默认 `3x256x256` 输入会产生 261 个 token，adapter 会在每个 sparse Linear 内部补齐到 32 倍数后切回原始 token 数。默认输出：

```text
artifacts/results/dinov3_vit7b16_cutlass_sparse_nvfp4/speed.csv
artifacts/results/dinov3_vit7b16_cutlass_sparse_nvfp4/accuracy.csv
```

Slurm 冒烟：

```bash
WARMUP=1 ITERS=2 sbatch scripts/slurm/bench_dinov3_vit7b16_cutlass_sparse_nvfp4_speed.sh
```

也可以复用已生成的结构化稀疏 checkpoint，避免转换时重新 magnitude prune：

```bash
CHECKPOINT=artifacts/checkpoints/dinov3_vit7b16/nvfp4_semi_structured_sparse/model.pt \
NO_PRUNE=1 WARMUP=1 ITERS=2 \
sbatch scripts/slurm/bench_dinov3_vit7b16_cutlass_sparse_nvfp4_speed.sh
```

CUTLASS sparse BF16 路径使用 cuSPARSELt 2:4 BF16 sparse kernel，不做 NVFP4 量化；推荐从已有
`semi_structured_sparse` checkpoint 导出 runtime checkpoint。adapter 会把 token 数补齐到 8 倍数。默认输出：

```text
artifacts/results/dinov3_vit7b16_cutlass_sparse_bf16/speed.csv
artifacts/results/dinov3_vit7b16_cutlass_sparse_bf16/accuracy.csv
```

Slurm 冒烟：

```bash
RUNTIME_CHECKPOINT=artifacts/checkpoints/dinov3_vit7b16/cutlass_sparse_bf16_runtime/model.pt \
WARMUP=1 ITERS=2 \
sbatch scripts/slurm/bench_dinov3_vit7b16_cutlass_sparse_bf16_speed.sh
```

如果要验证“从真实 CUTLASS packed checkpoint 加载后推理”，先在 GPU 节点导出 runtime-packed checkpoint。导出的 checkpoint 使用
`checkpoint_format=cutlass_runtime_packed_v1`，目标 Linear 不再保存 dense `.weight`，而是保存 CUTLASS dense
`packed_weight/weight_scale/weight_global_scale` 或 sparse `sparse_weight/metadata/weight_scale/weight_global_scale`。

```bash
BACKEND=all sbatch scripts/slurm/prepare_dinov3_cutlass_runtime_checkpoints.sh
```

导出后用 `RUNTIME_CHECKPOINT` 跑同一套 speed/accuracy 脚本：

```bash
RUNTIME_CHECKPOINT=artifacts/checkpoints/dinov3_vit7b16/cutlass_nvfp4_runtime/model.pt \
WARMUP=1 ITERS=2 \
sbatch scripts/slurm/bench_dinov3_vit7b16_cutlass_nvfp4_speed.sh

RUNTIME_CHECKPOINT=artifacts/checkpoints/dinov3_vit7b16/cutlass_nvfp4_runtime/model.pt \
sbatch scripts/slurm/eval_dinov3_vit7b16_cutlass_nvfp4_accuracy.sh

RUNTIME_CHECKPOINT=artifacts/checkpoints/dinov3_vit7b16/cutlass_sparse_nvfp4_runtime/model.pt \
WARMUP=1 ITERS=2 \
sbatch scripts/slurm/bench_dinov3_vit7b16_cutlass_sparse_nvfp4_speed.sh

RUNTIME_CHECKPOINT=artifacts/checkpoints/dinov3_vit7b16/cutlass_sparse_nvfp4_runtime/model.pt \
sbatch scripts/slurm/eval_dinov3_vit7b16_cutlass_sparse_nvfp4_accuracy.sh

RUNTIME_CHECKPOINT=artifacts/checkpoints/dinov3_vit7b16/cutlass_sparse_bf16_runtime/model.pt \
WARMUP=1 ITERS=2 \
sbatch scripts/slurm/bench_dinov3_vit7b16_cutlass_sparse_bf16_speed.sh

RUNTIME_CHECKPOINT=artifacts/checkpoints/dinov3_vit7b16/cutlass_sparse_bf16_runtime/model.pt \
sbatch scripts/slurm/eval_dinov3_vit7b16_cutlass_sparse_bf16_accuracy.sh
```

CSV 会额外记录 `checkpoint_format`、`runtime_checkpoint_path`、`source_checkpoint_path`、
`packed_checkpoint_file_size_bytes` 和 loader mode，便于和 on-the-fly CUTLASS 结果区分。

sparse NVFP4 还支持更省磁盘的 storage checkpoint。storage checkpoint 保存 compact pairwise 4:8 FP4 pairs，
不是 kernel-ready runtime buffer。speed/accuracy 可以直接传 `STORAGE_CHECKPOINT`，模型加载时会在内存中转换为
CUTLASS runtime buffers，不会重新 prune/quantize，也不会额外保存一份 runtime checkpoint。

```bash
sbatch scripts/slurm/prepare_dinov3_cutlass_storage_checkpoints.sh

STORAGE_CHECKPOINT=artifacts/checkpoints/dinov3_vit7b16/cutlass_sparse_nvfp4_storage/model.pt \
WARMUP=1 ITERS=2 \
sbatch scripts/slurm/bench_dinov3_vit7b16_cutlass_sparse_nvfp4_speed.sh

STORAGE_CHECKPOINT=artifacts/checkpoints/dinov3_vit7b16/cutlass_sparse_nvfp4_storage/model.pt \
sbatch scripts/slurm/eval_dinov3_vit7b16_cutlass_sparse_nvfp4_accuracy.sh
```

默认 sparse storage 输出：

```text
artifacts/checkpoints/dinov3_vit7b16/cutlass_sparse_nvfp4_storage/model.pt
artifacts/checkpoints/dinov3_vit7b16/cutlass_sparse_nvfp4_storage/metadata.json
```

如果希望缓存 kernel-ready runtime checkpoint，仍可显式从 storage 转 runtime：

```bash
SPARSE_STORAGE_CHECKPOINT=artifacts/checkpoints/dinov3_vit7b16/cutlass_sparse_nvfp4_storage/model.pt \
BACKEND=sparse_nvfp4 \
sbatch scripts/slurm/prepare_dinov3_cutlass_runtime_checkpoints.sh
```

## FlashInfer / Custom Shapes

- `bench_flashinfer_custom_shapes.py`：直接按自定义 `(m,n,k)` 测试 FlashInfer NVFP4 activation quant、FP4 GEMM、forward-like，以及 dense Linear bf16/fp32 baseline。
- `analyze_flashinfer_custom_shapes.py`：汇总 custom shape benchmark，生成 summary CSV、breakdown CSV、图表和 Markdown 报告。

默认输出：

```text
artifacts/analysis/flashinfer/custom_shapes.csv
artifacts/analysis/flashinfer/summary.md
```

快速冒烟：

```bash
python scripts/bench_flashinfer_custom_shapes.py --preset smoke --warmup 2 --iters 5
python scripts/analyze_flashinfer_custom_shapes.py
```

Slurm 冒烟：

```bash
PRESET=smoke WARMUP=2 ITERS=5 sbatch scripts/slurm/analysis/bench_flashinfer_custom_shapes.sh
```

## Slurm

作业脚本统一从项目根目录运行，并激活 `wja-cospaq`：

```bash
module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq
export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"
cd /data/home/scxj523/run/wja/project/my/fake/
```
