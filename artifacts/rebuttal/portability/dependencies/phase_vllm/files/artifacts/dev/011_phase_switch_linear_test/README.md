# Phase Switch Linear Test

This artifact tests offline vLLM generation with different Linear compression
methods for prefill and decode:

- prefill: `dense_nvfp4`
- decode: `w4a16_ours` / Marlin NVFP4

All code is artifact-local. The test runner imports `phase_switch_mytest.py`
before constructing `vllm.LLM`, registering the custom quantization method at
runtime without editing vLLM source files.

Run:

```bash
bash artifacts/dev/011_phase_switch_linear_test/run_smoke.sh
```

Outputs:

- `models/llama-2-7b-phase-switch-dense-nvfp4-marlin/`
- `results/generate_phase_switch.json`
- `results/switch_trace.json`
- `logs/export_phase_switch.log`
- `logs/generate_phase_switch.log`
