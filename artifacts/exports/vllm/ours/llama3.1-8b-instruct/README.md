# Llama-3.1-8B-Instruct vLLM Ours Max-Speed

This experiment produces one unconstrained predictor max-speed phase-heterogeneous policy for each workload:

- `prefill_only`: `batch=8,input=2048,output=1` (the policy itself has zero decode steps).
- `prefill_decode`: `batch=16,input=2048,output=80`.

The router uses real vLLM fused Linear shapes. In particular, Llama3.1 GQA fuses
`q_proj`, `k_proj`, and `v_proj` to a `6144x4096` QKV projection, not Llama2's
`12288x4096` shape. Predictor `marlin_nvfp4` is exported as runtime `w4a16_ours`.

All launchers default to GPUs 5--7 only. First create policies and checkpoints:

```bash
bash artifacts/exports/vllm/ours/llama3.1-8b-instruct/scripts/run_prepare.sh
```

Then run prefill-only speed, phase-switch fresh-process speed, full PMPD quality,
and the result summary with the respective scripts in `scripts/`.
