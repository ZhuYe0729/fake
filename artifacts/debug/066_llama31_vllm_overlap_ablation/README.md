# Llama3.1 vLLM overlap ablation

This artifact isolates the runtime benefit of overlapping decode-weight H2D
prefetch with prefill computation. It does not alter any main-experiment result.

## Fixed protocol

- Llama3.1-8B-Instruct
- solved max-speed policy `point_011` from experiment 057
- batch 8, input length 2048, output length 64
- BF16 KV cache, chunked prefill disabled, eager vLLM V1 execution
- one warm-up pair followed by five retained pairs on one GPU
- paired AB/BA order, with tracing disabled for formal timing

The policy is a useful runtime ablation point because all 128 Linear modules
change method between prefill and decode. The only experimental variable is
`PHASE_HETERO_SYNC_COPY`.

## Run

```bash
bash scripts/materialize_checkpoint.sh
/home/agent/wja/miniconda3/envs/cospaq/bin/python scripts/audit_config.py
/home/agent/wja/miniconda3/envs/cospaq/bin/python scripts/run_paired_latency.py --gpu 7
/home/agent/wja/miniconda3/envs/cospaq/bin/python scripts/summarize.py
```

The materialized checkpoint is reproducible and may be deleted after the raw
latency samples and report have been generated.

