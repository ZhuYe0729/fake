# FakeVLM Linear Time Proportion Study

This debug package measures how much FakeVLM dense BF16 inference time is spent in `nn.Linear`.

## Run

```bash
bash artifacts/debug/028_fakevlm_linear_time_proportion/scripts/launch_4gpu.sh
```

Defaults:

- Conda env: `cospaq`
- GPUs: `7 6 5 4`
- Model: `/home/agent/wja/data/models/lingcco/fakeVLM`
- Dataset: FakeClue test set paths used by existing FakeVLM experiments

Override examples:

```bash
GPUS="7 6" WARMUP=1 ITERS=3 bash artifacts/debug/028_fakevlm_linear_time_proportion/scripts/launch_4gpu.sh
WORKLOADS="prefill_b1_i1024 normal_01" bash artifacts/debug/028_fakevlm_linear_time_proportion/scripts/launch_4gpu.sh
```

## Outputs

- `results/fakevlm_linear_proportion_raw.csv`
- `summary/fakevlm_linear_proportion_summary.csv`
- `summary/analysis_report.md`
- `logs/*.log`
