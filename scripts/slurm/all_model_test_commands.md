# All Model Test Commands

默认避开节点 `wqd10nah09g4`。所有命令都在仓库根目录 `/data/home/scxj523/run/wja/project/my/fake/` 下执行。

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

# 精度
MODEL=maxvit MAXVIT_VARIANT=tiny METHOD=nvfp4 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=tiny METHOD=unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=tiny METHOD=semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=tiny METHOD=nvfp4_unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=tiny METHOD=nvfp4_semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh

# 速度
MODEL=maxvit MAXVIT_VARIANT=tiny METHOD=nvfp4 sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
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

# 精度
MODEL=maxvit MAXVIT_VARIANT=small METHOD=nvfp4 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=small METHOD=unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=small METHOD=semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=small METHOD=nvfp4_unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=small METHOD=nvfp4_semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh

# 速度
MODEL=maxvit MAXVIT_VARIANT=small METHOD=nvfp4 sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
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

# 精度
MODEL=maxvit MAXVIT_VARIANT=base METHOD=nvfp4 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=base METHOD=unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=base METHOD=semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=base METHOD=nvfp4_unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=base METHOD=nvfp4_semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh

# 速度
MODEL=maxvit MAXVIT_VARIANT=base METHOD=nvfp4 sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
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

# 精度
MODEL=maxvit MAXVIT_VARIANT=large METHOD=nvfp4 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=large METHOD=unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=large METHOD=semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=large METHOD=nvfp4_unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=maxvit MAXVIT_VARIANT=large METHOD=nvfp4_semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh

# 速度
MODEL=maxvit MAXVIT_VARIANT=large METHOD=nvfp4 sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=maxvit MAXVIT_VARIANT=large METHOD=unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=maxvit MAXVIT_VARIANT=large METHOD=semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=maxvit MAXVIT_VARIANT=large METHOD=nvfp4_unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=maxvit MAXVIT_VARIANT=large METHOD=nvfp4_semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
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

# 精度
MODEL=dinov3_vit7b16 METHOD=nvfp4 sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=dinov3_vit7b16 METHOD=unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=dinov3_vit7b16 METHOD=semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=dinov3_vit7b16 METHOD=nvfp4_unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
MODEL=dinov3_vit7b16 METHOD=nvfp4_semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh

# 速度
MODEL=dinov3_vit7b16 METHOD=nvfp4 sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=dinov3_vit7b16 METHOD=unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=dinov3_vit7b16 METHOD=semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=dinov3_vit7b16 METHOD=nvfp4_unstructured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
MODEL=dinov3_vit7b16 METHOD=nvfp4_semi_structured_sparse sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
```
