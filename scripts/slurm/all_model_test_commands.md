# All Model Test Commands

默认避开节点 `wqd10nah09g4`。所有命令都在仓库根目录 `/data/home/scxj523/run/wja/project/my/fake/` 下执行。

## MIRROR Dense 鉴伪评测

默认使用本地 MIRROR 权重 `/data/home/scxj523/run/wja/data/models/facebook/MIRROR/weight`，结果写入 `artifacts/results/mirror_dense/accuracy.csv`。

```shell
# Chameleon + GenImage validation
sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_mirror_dense_accuracy.sh

# Chameleon + GenImage validation，显式读取已解压 GenImage 目录
PREFER_EXTRACTED_GENIMAGE=1 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_mirror_dense_accuracy.sh

# 只评测 Chameleon
BENCHMARKS=Chameleon sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_mirror_dense_accuracy.sh

# 只评测 GenImage validation
BENCHMARKS=GenImage sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_mirror_dense_accuracy.sh

# 小样本 smoke test
LIMIT_PER_CLASS=8 BENCHMARKS="Chameleon GenImage" sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_mirror_dense_accuracy.sh
```

## MIRROR Compressed 鉴伪评测

压缩 checkpoint 默认写入 `artifacts/checkpoints/mirror/{method}/model.pt`，精度写入 `artifacts/results/mirror_compressed/accuracy.csv`，速度写入 `artifacts/results/mirror_compressed/speed.csv`。速度只跑真实 runtime 方法：`dense`、`nvfp4`、`semi_structured_sparse`、`nvfp4_semi_structured_sparse`。

```shell
# 准备全部 MIRROR 压缩 checkpoint
sbatch --exclude=wqd10nah09g4 scripts/slurm/prepare_mirror_compressed_models.sh

# smoke checkpoint 准备
METHODS="nvfp4" LIMIT_PER_CLASS=8 sbatch --exclude=wqd10nah09g4 scripts/slurm/prepare_mirror_compressed_models.sh

# 全部方法精度
METHODS="dense nvfp4 int4 unstructured_sparse semi_structured_sparse nvfp4_unstructured_sparse nvfp4_semi_structured_sparse int4_unstructured_sparse int4_semi_structured_sparse nvfp4_4over6_unstructured_sparse nvfp4_4over6_semi_structured_sparse" \
sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_mirror_compressed_accuracy.sh

# 单方法精度
METHOD=nvfp4 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_mirror_compressed_accuracy.sh

# 小样本 smoke 精度
METHODS="dense nvfp4" LIMIT_PER_CLASS=8 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_mirror_compressed_accuracy.sh

# 真实 runtime 速度
sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_mirror_compressed_speed.sh

# 单方法速度
METHOD=nvfp4 sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_mirror_compressed_speed.sh

# 按 DINO batch sweep 口径测试真实 runtime 多 batch size，输出 speed_batch_sweep.csv
sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_mirror_batch_sweep_speed.sh

# 只跑部分方法/部分 batch size
METHODS="dense semi_structured_sparse nvfp4" BATCH_SIZES="1 2 4 8 16" \
sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_mirror_batch_sweep_speed.sh
```

## MaxViT Dense Baseline

### tiny

```shell
MAXVIT_VARIANT=tiny sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_maxvit_dense_accuracy.sh
MAXVIT_VARIANT=tiny sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_maxvit_dense_speed.sh
```

### small

```shell
MAXVIT_VARIANT=small sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_maxvit_dense_accuracy.sh
MAXVIT_VARIANT=small sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_maxvit_dense_speed.sh
```

### base

```shell
MAXVIT_VARIANT=base sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_maxvit_dense_accuracy.sh
MAXVIT_VARIANT=base sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_maxvit_dense_speed.sh
```

### large

```shell
MAXVIT_VARIANT=large sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_maxvit_dense_accuracy.sh
MAXVIT_VARIANT=large sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_maxvit_dense_speed.sh
```

## MaxViT Compressed

### tiny

```shell
# 模型压缩准备
MODEL=maxvit MAXVIT_VARIANT=tiny METHODS="nvfp4 unstructured_sparse semi_structured_sparse nvfp4_unstructured_sparse nvfp4_semi_structured_sparse" \
sbatch --exclude=wqd10nah09g4 scripts/slurm/prepare_compressed_models.sh

# 纯 INT4
MODEL=maxvit MAXVIT_VARIANT=tiny METHODS="int4" \
sbatch --exclude=wqd10nah09g4 scripts/slurm/prepare_compressed_models.sh

# INT4 + SparseGPT opt-in
MODEL=maxvit MAXVIT_VARIANT=tiny METHODS="int4_unstructured_sparse int4_semi_structured_sparse" \
sbatch --exclude=wqd10nah09g4 scripts/slurm/prepare_compressed_models.sh

# 精度
MODEL=maxvit MAXVIT_VARIANT=tiny METHOD=nvfp4 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=tiny METHOD=int4 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=tiny METHOD=unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=tiny METHOD=semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=tiny METHOD=nvfp4_unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=tiny METHOD=nvfp4_semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=tiny METHOD=int4_unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=tiny METHOD=int4_semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh

# WA fake 精度，写入 artifacts/results/maxvit_tiny_compressed/accuracy_wa_fake.csv
MODEL=maxvit MAXVIT_VARIANT=tiny METHOD=nvfp4_unstructured_sparse WA_FAKE=1 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=tiny METHOD=int4 WA_FAKE=1 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=tiny METHOD=int4_unstructured_sparse WA_FAKE=1 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=tiny METHOD=int4_semi_structured_sparse WA_FAKE=1 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh

# 速度
MODEL=maxvit MAXVIT_VARIANT=tiny METHOD=nvfp4 sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=maxvit MAXVIT_VARIANT=tiny METHOD=int4 sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=maxvit MAXVIT_VARIANT=tiny METHOD=unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=maxvit MAXVIT_VARIANT=tiny METHOD=semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=maxvit MAXVIT_VARIANT=tiny METHOD=nvfp4_unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=maxvit MAXVIT_VARIANT=tiny METHOD=nvfp4_semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
```

### small

```shell
# 模型压缩准备
MODEL=maxvit MAXVIT_VARIANT=small METHODS="nvfp4 unstructured_sparse semi_structured_sparse nvfp4_unstructured_sparse nvfp4_semi_structured_sparse" \
sbatch --exclude=wqd10nah09g4 scripts/slurm/prepare_compressed_models.sh

# 纯 INT4
MODEL=maxvit MAXVIT_VARIANT=small METHODS="int4" \
sbatch --exclude=wqd10nah09g4 scripts/slurm/prepare_compressed_models.sh

# 精度
MODEL=maxvit MAXVIT_VARIANT=small METHOD=nvfp4 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=small METHOD=int4 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=small METHOD=unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=small METHOD=semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=small METHOD=nvfp4_unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=small METHOD=nvfp4_semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh

# WA fake 精度，写入 artifacts/results/maxvit_small_compressed/accuracy_wa_fake.csv
MODEL=maxvit MAXVIT_VARIANT=small METHOD=nvfp4_unstructured_sparse WA_FAKE=1 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=small METHOD=int4 WA_FAKE=1 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=small METHOD=int4_unstructured_sparse WA_FAKE=1 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=small METHOD=int4_semi_structured_sparse WA_FAKE=1 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh

# 速度
MODEL=maxvit MAXVIT_VARIANT=small METHOD=nvfp4 sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=maxvit MAXVIT_VARIANT=small METHOD=int4 sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=maxvit MAXVIT_VARIANT=small METHOD=unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=maxvit MAXVIT_VARIANT=small METHOD=semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=maxvit MAXVIT_VARIANT=small METHOD=nvfp4_unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=maxvit MAXVIT_VARIANT=small METHOD=nvfp4_semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
```

### base

```shell
# 模型压缩准备
MODEL=maxvit MAXVIT_VARIANT=base METHODS="nvfp4 unstructured_sparse semi_structured_sparse nvfp4_unstructured_sparse nvfp4_semi_structured_sparse" \
sbatch --exclude=wqd10nah09g4 scripts/slurm/prepare_compressed_models.sh

# 纯 INT4
MODEL=maxvit MAXVIT_VARIANT=base METHODS="int4" \
sbatch --exclude=wqd10nah09g4 scripts/slurm/prepare_compressed_models.sh

# 精度
MODEL=maxvit MAXVIT_VARIANT=base METHOD=nvfp4 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=base METHOD=int4 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=base METHOD=unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=base METHOD=semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=base METHOD=nvfp4_unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=base METHOD=nvfp4_semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh

# WA fake 精度，写入 artifacts/results/maxvit_base_compressed/accuracy_wa_fake.csv
MODEL=maxvit MAXVIT_VARIANT=base METHOD=nvfp4_unstructured_sparse WA_FAKE=1 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=base METHOD=int4 WA_FAKE=1 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=base METHOD=int4_unstructured_sparse WA_FAKE=1 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=base METHOD=int4_semi_structured_sparse WA_FAKE=1 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh

# 速度
MODEL=maxvit MAXVIT_VARIANT=base METHOD=nvfp4 sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=maxvit MAXVIT_VARIANT=base METHOD=int4 sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=maxvit MAXVIT_VARIANT=base METHOD=unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=maxvit MAXVIT_VARIANT=base METHOD=semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=maxvit MAXVIT_VARIANT=base METHOD=nvfp4_unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=maxvit MAXVIT_VARIANT=base METHOD=nvfp4_semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
```

### large

```shell
# 模型压缩准备
MODEL=maxvit MAXVIT_VARIANT=large METHODS="nvfp4 unstructured_sparse semi_structured_sparse nvfp4_unstructured_sparse nvfp4_semi_structured_sparse" \
sbatch --exclude=wqd10nah09g4 scripts/slurm/prepare_compressed_models.sh

# 纯 INT4
MODEL=maxvit MAXVIT_VARIANT=large METHODS="int4" \
sbatch --exclude=wqd10nah09g4 scripts/slurm/prepare_compressed_models.sh

# 精度
MODEL=maxvit MAXVIT_VARIANT=large METHOD=nvfp4 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=large METHOD=int4 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=large METHOD=unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=large METHOD=semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=large METHOD=nvfp4_unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=large METHOD=nvfp4_semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh

# WA fake 精度，写入 artifacts/results/maxvit_large_compressed/accuracy_wa_fake.csv
MODEL=maxvit MAXVIT_VARIANT=large METHOD=nvfp4_unstructured_sparse WA_FAKE=1 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=large METHOD=int4 WA_FAKE=1 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=large METHOD=int4_unstructured_sparse WA_FAKE=1 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=large METHOD=int4_semi_structured_sparse WA_FAKE=1 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh

# 速度
MODEL=maxvit MAXVIT_VARIANT=large METHOD=nvfp4 sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=maxvit MAXVIT_VARIANT=large METHOD=int4 sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=maxvit MAXVIT_VARIANT=large METHOD=unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=maxvit MAXVIT_VARIANT=large METHOD=semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=maxvit MAXVIT_VARIANT=large METHOD=nvfp4_unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=maxvit MAXVIT_VARIANT=large METHOD=nvfp4_semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
```

## MaxViT Four Over Six Fake Quant

> MaxViT 4/6 accuracy/speed 脚本默认开启 activation fake quant；设置 `NO_ACTIVATION_QUANT=1` 可关闭。

### 一次性准备全部 MaxViT 4/6 checkpoint

```shell
MAXVIT_VARIANTS="tiny small base large" \
METHODS="nvfp4_4over6_unstructured_sparse nvfp4_4over6_semi_structured_sparse" \
sbatch --exclude=wqd10nah09g4 scripts/slurm/prepare_maxvit_four_over_six_checkpoints.sh
```

### 一次性提交全部 MaxViT 4/6 精度/速度

```shell
# 默认：开启 activation fake quant
for v in tiny small base large; do
  for m in nvfp4_4over6_unstructured_sparse nvfp4_4over6_semi_structured_sparse; do
    MAXVIT_VARIANT="$v" METHOD="$m" sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_maxvit_four_over_six_accuracy.sh
    MAXVIT_VARIANT="$v" METHOD="$m" sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_maxvit_four_over_six_speed.sh
  done
done

# 关闭 activation fake quant
for v in tiny small base large; do
  for m in nvfp4_4over6_unstructured_sparse nvfp4_4over6_semi_structured_sparse; do
    MAXVIT_VARIANT="$v" METHOD="$m" NO_ACTIVATION_QUANT=1 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_maxvit_four_over_six_accuracy.sh
    MAXVIT_VARIANT="$v" METHOD="$m" NO_ACTIVATION_QUANT=1 sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_maxvit_four_over_six_speed.sh
  done
done
```

### tiny

```shell
# checkpoint 准备
MAXVIT_VARIANTS=tiny METHODS="nvfp4_4over6_unstructured_sparse nvfp4_4over6_semi_structured_sparse" \
sbatch --exclude=wqd10nah09g4 scripts/slurm/prepare_maxvit_four_over_six_checkpoints.sh

# 精度
MAXVIT_VARIANT=tiny METHOD=nvfp4_4over6_unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_maxvit_four_over_six_accuracy.sh
MAXVIT_VARIANT=tiny METHOD=nvfp4_4over6_semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_maxvit_four_over_six_accuracy.sh

# 速度
MAXVIT_VARIANT=tiny METHOD=nvfp4_4over6_unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_maxvit_four_over_six_speed.sh
MAXVIT_VARIANT=tiny METHOD=nvfp4_4over6_semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_maxvit_four_over_six_speed.sh
```

### small

```shell
# checkpoint 准备
MAXVIT_VARIANTS=small METHODS="nvfp4_4over6_unstructured_sparse nvfp4_4over6_semi_structured_sparse" \
sbatch --exclude=wqd10nah09g4 scripts/slurm/prepare_maxvit_four_over_six_checkpoints.sh

# 精度
MAXVIT_VARIANT=small METHOD=nvfp4_4over6_unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_maxvit_four_over_six_accuracy.sh
MAXVIT_VARIANT=small METHOD=nvfp4_4over6_semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_maxvit_four_over_six_accuracy.sh

# 速度
MAXVIT_VARIANT=small METHOD=nvfp4_4over6_unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_maxvit_four_over_six_speed.sh
MAXVIT_VARIANT=small METHOD=nvfp4_4over6_semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_maxvit_four_over_six_speed.sh
```

### base

```shell
# checkpoint 准备
MAXVIT_VARIANTS=base METHODS="nvfp4_4over6_unstructured_sparse nvfp4_4over6_semi_structured_sparse" \
sbatch --exclude=wqd10nah09g4 scripts/slurm/prepare_maxvit_four_over_six_checkpoints.sh

# 精度
MAXVIT_VARIANT=base METHOD=nvfp4_4over6_unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_maxvit_four_over_six_accuracy.sh
MAXVIT_VARIANT=base METHOD=nvfp4_4over6_semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_maxvit_four_over_six_accuracy.sh

# 速度
MAXVIT_VARIANT=base METHOD=nvfp4_4over6_unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_maxvit_four_over_six_speed.sh
MAXVIT_VARIANT=base METHOD=nvfp4_4over6_semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_maxvit_four_over_six_speed.sh
```

### large

```shell
# checkpoint 准备
MAXVIT_VARIANTS=large METHODS="nvfp4_4over6_unstructured_sparse nvfp4_4over6_semi_structured_sparse" \
sbatch --exclude=wqd10nah09g4 scripts/slurm/prepare_maxvit_four_over_six_checkpoints.sh

# 精度
MAXVIT_VARIANT=large METHOD=nvfp4_4over6_unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_maxvit_four_over_six_accuracy.sh
MAXVIT_VARIANT=large METHOD=nvfp4_4over6_semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_maxvit_four_over_six_accuracy.sh

# 速度
MAXVIT_VARIANT=large METHOD=nvfp4_4over6_unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_maxvit_four_over_six_speed.sh
MAXVIT_VARIANT=large METHOD=nvfp4_4over6_semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_maxvit_four_over_six_speed.sh
```

## DINOv3 ViT-7B Dense Baseline

```shell
sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_dinov3_vit7b16_dense_accuracy.sh
sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_dinov3_vit7b16_dense_speed.sh
```

## DINOv3 ViT-7B Compressed

```shell
# 模型压缩准备
MODEL=dinov3_vit7b16 METHODS="nvfp4 unstructured_sparse semi_structured_sparse nvfp4_unstructured_sparse nvfp4_semi_structured_sparse" \
sbatch --exclude=wqd10nah09g4 scripts/slurm/prepare_compressed_models.sh

# 纯 INT4
MODEL=dinov3_vit7b16 METHODS="int4" \
sbatch --exclude=wqd10nah09g4 scripts/slurm/prepare_compressed_models.sh

# 精度
MODEL=dinov3_vit7b16 METHOD=nvfp4 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=dinov3_vit7b16 METHOD=int4 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=dinov3_vit7b16 METHOD=unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=dinov3_vit7b16 METHOD=semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=dinov3_vit7b16 METHOD=nvfp4_unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=dinov3_vit7b16 METHOD=nvfp4_semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh

# WA fake 精度，写入 artifacts/results/dinov3_vit7b16_compressed/accuracy_wa_fake.csv
MODEL=dinov3_vit7b16 METHOD=nvfp4_unstructured_sparse WA_FAKE=1 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=dinov3_vit7b16 METHOD=int4 WA_FAKE=1 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=dinov3_vit7b16 METHOD=int4_unstructured_sparse WA_FAKE=1 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=dinov3_vit7b16 METHOD=int4_semi_structured_sparse WA_FAKE=1 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh

# 速度
MODEL=dinov3_vit7b16 METHOD=nvfp4 sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=dinov3_vit7b16 METHOD=int4 sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=dinov3_vit7b16 METHOD=unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=dinov3_vit7b16 METHOD=semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=dinov3_vit7b16 METHOD=nvfp4_unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=dinov3_vit7b16 METHOD=nvfp4_semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
```
