# Llama2-7B 4090 Prefill/Decode Speed

This debug package benchmarks Llama2-7B speed on the supercomputer 4090 partition across prefill-only, decode-heavy, and mixed prefill+decode scenarios.

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
- Scenarios:
  - `prefill_only`: `batch_size=16`, `input_tokens=1024`, `output_tokens=0`
  - `decode_heavy`: `batch_size=1`, `input_tokens=1024`, `output_tokens=256`
  - `prefill_decode`: `batch_size=1`, `input_tokens=16384`, `output_tokens=32`
- Methods: `dense_bf16`, `sparse_bf16`, `marlin_nvfp4`

## Outputs

- `results/full_model_summary.csv`: combined full-model latency and speedup summary across all scenarios.
- `results/full_model_raw.csv`: combined per-iteration full-model timings.
- `results/linear_summary.csv`: combined supporting linear aggregate latency across all scenarios.
- `results/<scenario>/full_model_summary.csv`: per-scenario full-model summary.
- `results/<scenario>/linear_summary.csv`: per-scenario linear aggregate summary.
- `results/full_model_prefill_summary.csv`: legacy compatibility copy for `prefill_only`.
- `results/linear_prefill_summary.csv`: legacy compatibility copy for `prefill_only`.
- `results/method_policies/`: generated uniform policies for compressed methods.
- `summary/README.md`: generated run summary.

`marlin_nvfp4` is the repo backend used for Marlin W4A16.

To run only selected scenarios:

```bash
SCENARIOS="decode_heavy prefill_decode" \
sbatch artifacts/debug/025_llama2_4090_prefill_speed/run_llama2_4090_prefill_speed_4090.sh
```

For a custom one-off scenario, call the Python script directly with `--custom-scenario --batch-size ... --input-tokens ... --output-tokens ...`.
