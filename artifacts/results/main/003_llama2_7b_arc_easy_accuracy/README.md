# Llama-2-7B Arc-Easy Real Compression Accuracy

This experiment prepares compressed Llama-2-7B artifacts with WikiText-2 calibration, then evaluates zero-shot `arc_easy` accuracy with real runtime kernels.

Default calibration:

- dataset: `wikitext/wikitext-2-raw-v1`, train split
- cache: `/home/agent/wja/.cache/huggingface`
- samples: `128`
- sequence length: `2048`
- seed: `0`

Methods:

- `dense_bf16`
- `sparse_bf16`
- `dense_nvfp4`
- `sparse_nvfp4`
- `marlin_nvfp4`
- `dense_nvfp4_prefill_marlin_decode`

The hybrid method reuses the prepared `dense_nvfp4` artifact.

Example smoke run:

```bash
python artifacts/results/main/003_llama2_7b_arc_easy_accuracy/scripts/launch.py \
  --phase all --calib-samples 2 --seq-len 128 --limit 1
```

Full run:

```bash
python artifacts/results/main/003_llama2_7b_arc_easy_accuracy/scripts/launch.py --phase all
```

The launcher checks physical GPUs `[7, 6, 5, 4, 3, 2]` by default, skips cards above `--max-used-mb`, sets `CUDA_VISIBLE_DEVICES` per subprocess, and leaves cards `0` and `1` untouched.

Smoke outputs are not official results. Run the full command without `--limit` and without `--skip-existing` to overwrite smoke artifacts with the intended `128 x 2048` WikiText-2 calibration and full `arc_easy` evaluation.

Outputs:

- `prepared/{method}/model.pt`
- `prepared/{method}/metadata.json`
- `prepared/{method}/compression_log.jsonl`
- `prepared/{method}/stdout.log` when launched through `launch.py`
- `methods/{method}/accuracy.json`
- `methods/{method}/eval_metadata.json`
- `methods/{method}/stdout.log` when launched through `launch.py`
- `summary/accuracy_summary.csv`
