# Llama2-7B 4090 Prefill Speed

This debug package benchmarks Llama2-7B prefill-only speed on the supercomputer 4090 partition.

## Run

From the project root on the supercomputer:

```bash
sbatch artifacts/debug/025_llama2_4090_prefill_speed/run_llama2_4090_prefill_speed_4090.sh
```

Default runtime settings:

- Partition: `gpu_4090`
- Conda env: `wja-cospaq`
- Project path: `/data/home/scxj523/run/wja/project/my/fake`
- Model path: `/data/home/scxj523/run/wja/data/models/LLM-Research/llama-2-7b`
- Scenario: `batch_size=16`, `input_tokens=1024`, `output_tokens=0`
- Methods: `dense_bf16`, `sparse_bf16`, `marlin_nvfp4`

## Outputs

- `results/full_model_prefill_raw.csv`: one row per measured full-model iteration.
- `results/full_model_prefill_summary.csv`: primary full-model latency and speedup summary.
- `results/linear_prefill_summary.csv`: supporting per-method linear aggregate latency.
- `results/method_policies/`: generated uniform policies for compressed methods.
- `summary/README.md`: generated run summary.

`marlin_nvfp4` is the repo backend used for Marlin W4A16.
