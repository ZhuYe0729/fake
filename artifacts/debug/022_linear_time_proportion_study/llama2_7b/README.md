# Llama2-7B Linear Time Proportion Study

This package extends the 022 linear time proportion study with a Llama2-7B dense BF16 run.
It is intentionally isolated from the existing Qwen3.5 results in the parent directory.

## Run

From the repo root:

```bash
bash artifacts/debug/022_linear_time_proportion_study/llama2_7b/run_parallel.sh
```

Defaults:

- Conda env: `cospaq`
- GPUs: `7,6,5,4`
- Model path resolution order:
  - `$LLAMA2_MODEL_PATH`
  - `/data/home/scxj523/run/wja/data/models/LLM-Research/llama-2-7b`
  - `/home/agent/wja/data/models/LLM-Research/llama-2-7b`
- Batch sizes: `1,4,16,32,64`
- Input tokens: `16,64,256,1024,4096,8192`
- Speed output tokens: `1,32,128,256`
- Breakdown output tokens: `1,32`

Each GPU runs exactly one worker process per phase. Workers shard the config list round-robin.

## Outputs

- `speed/llama2_7b_speed_shard*.csv`
- `breakdown_coarse/llama2_7b_breakdown_coarse_shard*.csv`
- `logs/*.log`
- `summary/llama2_linear_proportion_summary.csv`
- `summary/analysis_report.md`
- `summary/qwen_vs_llama_context.md`

OOM and failed configs are written into the CSV with a `status` field.
