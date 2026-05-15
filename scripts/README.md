# Scripts

这个目录放项目级的可执行脚本。后续新增脚本时，在这里补一条简短说明即可；复杂实验细节放到对应的 `dev/plans/` 和 `dev/impls/` 记录里。

大多数 benchmark/eval 脚本需要在 GPU 计算节点运行。登录节点通常只适合做语法检查、准备文件和提交 Slurm 作业。

## MaxViT / NVFP4

- `bench_maxvit_dense_speed.py`：MaxViT dense 端到端 forward benchmark。
- `bench_maxvit_nvfp4_speed.py`：MaxViT FlashInfer NVFP4 端到端 forward benchmark。
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

## DINOv3 / NVFP4

- `bench_dinov3_vit7b16_dense_speed.py`：DINOv3 ViT-7B/16 dense classifier forward benchmark。
- `bench_dinov3_vit7b16_nvfp4_micro.py`：逐层拆解 DINOv3 ViT-7B/16 backbone NVFP4 Linear 的耗时。
- `bench_dinov3_vit7b16_cutlass_nvfp4_speed.py`：DINOv3 ViT-7B/16 CUTLASS dense NVFP4 classifier forward benchmark，使用真实 CUTLASS NVFP4 kernel。
- `eval_dinov3_vit7b16_cutlass_nvfp4_accuracy.py`：DINOv3 ViT-7B/16 CUTLASS dense NVFP4 ImageNet linear-head accuracy。

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
