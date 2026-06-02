# Qwen3.5 Other Variants Real-Kernel Test Commands

Assumptions:
- Card 0 is occupied, so commands start from physical GPU card 1.
- Run each model block in a separate tmux pane/session.
- 2B/4B/9B each reserves 1 GPU; 27B reserves 4 GPUs via `CUDA_VISIBLE_DEVICES=4,5,6,7`.
- This is for a local machine, not the supercomputer environment; use conda env `cospaq`.
- 27B uses `--device-map auto`; `--max-memory` indices are relative to `CUDA_VISIBLE_DEVICES`, so `0:30GiB` means physical card 4 in the 27B block below.
- Benchmark grid keeps the previous output token settings and adds batch sizes 16/32 plus input tokens 8192:
  - batch sizes: `1 2 4 8 16 32`
  - input tokens: `128 512 1024 2048 8192`
  - output tokens: `32 128`

Common methods:
- prepare methods: `dense_nvfp4 sparse_bf16 sparse_nvfp4 marlin_nvfp4`
- benchmark methods: `dense dense_nvfp4 sparse_bf16 sparse_nvfp4 marlin_nvfp4`

## tmux 1: Qwen3.5-2B on card 1

```bash
cd /root/wja/project/my/cospaq/fake
conda activate cospaq

export CUDA_VISIBLE_DEVICES=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.

VARIANT=2B
METHODS_PREPARE="dense_nvfp4 sparse_bf16 sparse_nvfp4 marlin_nvfp4"
METHODS_BENCH="dense dense_nvfp4 sparse_bf16 sparse_nvfp4 marlin_nvfp4"
BATCH_SIZES="1 2 4 8 16 32"
INPUT_TOKENS="128 512 1024 2048 8192"
OUTPUT_TOKENS="32 128"
WARMUP=5
ITERS=20

for METHOD in $METHODS_PREPARE; do
  python scripts/prepare_qwen3_5_kernel_checkpoint.py \
    --variant "$VARIANT" \
    --method "$METHOD" \
    --dtype bf16
done

for METHOD in $METHODS_BENCH; do
  python scripts/bench_qwen3_5_speed.py \
    --variant "$VARIANT" \
    --method "$METHOD" \
    --batch-sizes $BATCH_SIZES \
    --input-tokens $INPUT_TOKENS \
    --output-tokens $OUTPUT_TOKENS \
    --warmup "$WARMUP" \
    --iters "$ITERS" \
    --output-csv "artifacts/results/qwen3_5_2b_${METHOD}/speed.csv" \
    --verbose
done
```

## tmux 2: Qwen3.5-4B on card 2

```bash
cd /root/wja/project/my/cospaq/fake
conda activate cospaq

export CUDA_VISIBLE_DEVICES=2
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.

VARIANT=4B
METHODS_PREPARE="dense_nvfp4 sparse_bf16 sparse_nvfp4 marlin_nvfp4"
METHODS_BENCH="dense dense_nvfp4 sparse_bf16 sparse_nvfp4 marlin_nvfp4"
BATCH_SIZES="1 2 4 8 16 32"
INPUT_TOKENS="128 512 1024 2048 8192"
OUTPUT_TOKENS="32 128"
WARMUP=5
ITERS=20

for METHOD in $METHODS_PREPARE; do
  python scripts/prepare_qwen3_5_kernel_checkpoint.py \
    --variant "$VARIANT" \
    --method "$METHOD" \
    --dtype bf16
done

for METHOD in $METHODS_BENCH; do
  python scripts/bench_qwen3_5_speed.py \
    --variant "$VARIANT" \
    --method "$METHOD" \
    --batch-sizes $BATCH_SIZES \
    --input-tokens $INPUT_TOKENS \
    --output-tokens $OUTPUT_TOKENS \
    --warmup "$WARMUP" \
    --iters "$ITERS" \
    --output-csv "artifacts/results/qwen3_5_4b_${METHOD}/speed.csv" \
    --verbose
done
```

## tmux 3: Qwen3.5-9B on card 3

```bash
cd /root/wja/project/my/cospaq/fake
conda activate cospaq

export CUDA_VISIBLE_DEVICES=3
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.

VARIANT=9B
METHODS_PREPARE="dense_nvfp4 sparse_bf16 sparse_nvfp4 marlin_nvfp4"
METHODS_BENCH="dense dense_nvfp4 sparse_bf16 sparse_nvfp4 marlin_nvfp4"
BATCH_SIZES="1 2 4 8 16 32"
INPUT_TOKENS="128 512 1024 2048 8192"
OUTPUT_TOKENS="32 128"
WARMUP=5
ITERS=20

for METHOD in $METHODS_PREPARE; do
  python scripts/prepare_qwen3_5_kernel_checkpoint.py \
    --variant "$VARIANT" \
    --method "$METHOD" \
    --dtype bf16
done

for METHOD in $METHODS_BENCH; do
  python scripts/bench_qwen3_5_speed.py \
    --variant "$VARIANT" \
    --method "$METHOD" \
    --batch-sizes $BATCH_SIZES \
    --input-tokens $INPUT_TOKENS \
    --output-tokens $OUTPUT_TOKENS \
    --warmup "$WARMUP" \
    --iters "$ITERS" \
    --output-csv "artifacts/results/qwen3_5_9b_${METHOD}/speed.csv" \
    --verbose
done
```

## tmux 4: Qwen3.5-27B on cards 4, 5, 6 and 7

```bash
cd /root/wja/project/my/cospaq/fake
conda activate cospaq

export CUDA_VISIBLE_DEVICES=4,5,6,7
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.

VARIANT=27B
METHODS_PREPARE="dense_nvfp4 sparse_bf16 sparse_nvfp4 marlin_nvfp4"
METHODS_BENCH="dense dense_nvfp4 sparse_bf16 sparse_nvfp4 marlin_nvfp4"
BATCH_SIZES="1 2 4 8 16 32"
INPUT_TOKENS="128 512 1024 2048 8192"
OUTPUT_TOKENS="32 128"
WARMUP=5
ITERS=20
DEVICE_MAP_ARGS="--device-map auto --max-memory 0:30GiB 1:30GiB 2:30GiB 3:30GiB"

for METHOD in $METHODS_PREPARE; do
  python scripts/prepare_qwen3_5_kernel_checkpoint.py \
    --variant "$VARIANT" \
    --method "$METHOD" \
    --dtype bf16 \
    $DEVICE_MAP_ARGS
done

for METHOD in $METHODS_BENCH; do
  python scripts/bench_qwen3_5_speed.py \
    --variant "$VARIANT" \
    --method "$METHOD" \
    --batch-sizes $BATCH_SIZES \
    --input-tokens $INPUT_TOKENS \
    --output-tokens $OUTPUT_TOKENS \
    --warmup "$WARMUP" \
    --iters "$ITERS" \
    --output-csv "artifacts/results/qwen3_5_27b_${METHOD}/speed.csv" \
    $DEVICE_MAP_ARGS \
    --verbose
done
```
