# FakeVLM Uniform Accuracy Debug

This directory contains the debug-only scripts and outputs for FakeVLM uniform compression accuracy.

## Methods

- `dense_bf16`
- `sparse_bf16`
- `dense_nvfp4`
- `sparse_nvfp4`
- `marlin_weight_only`
- `dense_nvfp4_prefill_marlin_decode`

The NVFP4 W4A4 methods use the real runtime path with online dynamic activation global scale and activation quantization. Sparse methods first run calibration over FakeClue samples to collect activation Hessian/importance, prune the dense weights, then install the runtime wrapper with wrapper-side pruning disabled.

## Defaults

- Conda env: `cospaq`
- Model: `/home/agent/wja/data/models/lingcco/fakeVLM`
- Test JSON: `/home/agent/wja/data/datasets/lingcco/FakeClue/data_json/test.json`
- Image root: `/home/agent/wja/data/datasets/lingcco/FakeClue/test/test`
- GPU order: `7,6,5,4,3,2`
- GPU `0` and `1` are intentionally unused.
- Hybrid decode threshold: `m <= 8`

## Commands

Smoke:

```bash
bash artifacts/debug/020_fakevlm_uniform_accuracy/run_smoke_parallel.sh
```

Full accuracy:

```bash
bash artifacts/debug/020_fakevlm_uniform_accuracy/run_accuracy_parallel.sh
```

Useful overrides:

```bash
BATCH_SIZE=8 WORKERS=2 OVERWRITE=1 bash artifacts/debug/020_fakevlm_uniform_accuracy/run_accuracy_parallel.sh
```

## Outputs

- `configs/`: run configuration and environment metadata.
- `compression/<method>/`: selected modules and compression metadata.
- `logs/<method>.log`: per-method stdout/stderr.
- `status/<method>.json`: per-method process status.
- `outputs/<method>/predictions.json`: per-sample generations.
- `outputs/<method>/accuracy.json`: aggregate accuracy.
- `summary/accuracy_summary.csv`: cross-method summary.

## Full Accuracy Results

Run command:

```bash
BATCH_SIZE=8 WORKERS=2 OVERWRITE=1 bash artifacts/debug/020_fakevlm_uniform_accuracy/run_accuracy_parallel.sh
```

All methods completed on 5000 samples with one process per GPU in physical GPU order `7,6,5,4,3,2`.

| Method | GPU | Accuracy | Right | Wrong | Replaced Linear | Activation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `dense_bf16` | 7 | 0.9864 | 4932 | 68 | 0 | none |
| `sparse_bf16` | 6 | 0.9852 | 4926 | 74 | 224 | none |
| `dense_nvfp4` | 5 | 0.9870 | 4935 | 65 | 224 | dynamic online NVFP4 |
| `sparse_nvfp4` | 4 | 0.7686 | 3843 | 1157 | 224 | dynamic online NVFP4 |
| `marlin_weight_only` | 3 | 0.9876 | 4938 | 62 | 224 | BF16 |
| `dense_nvfp4_prefill_marlin_decode` | 2 | 0.9868 | 4934 | 66 | 224 | prefill W4A4 online, decode W4A16 |

## TODO

Speed is intentionally out of scope for this debug run. See `TODO_speed.md`.
