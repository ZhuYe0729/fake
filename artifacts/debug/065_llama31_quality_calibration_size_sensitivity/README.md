# Llama3.1 quality calibration-size sensitivity

This experiment refits the unchanged 064 full phase-aware quality surrogate on
nested calibration sets. It combines the frozen 057 54-policy training set and
18 calibration-only extensions into one exchangeable 72-policy candidate pool.
The fixed 18-policy holdout is never used for policy generation, subset
construction, or model fitting. The particular 54-policy model used by the main
experiments remains frozen and is plotted separately from the candidate-pool
sensitivity curve.

Generate the deterministic extension policies with:

```bash
python \
  artifacts/debug/065_llama31_quality_calibration_size_sensitivity/scripts/generate_extension_policies.py
```

Collect the 18 additional real phase-vLLM NLL labels (resumable) with:

```bash
/home/agent/wja/miniconda3/envs/vllm/bin/python \
  artifacts/debug/065_llama31_quality_calibration_size_sensitivity/scripts/run_extension_nll.py \
  --gpus 1,2,3,4,5,6,7

python \
  artifacts/debug/065_llama31_quality_calibration_size_sensitivity/scripts/merge_extension_nll.py
```

`--gpu-memory-utilization` may be reduced from 0.80 to 0.75 or 0.70 for
policies with a larger materialization peak. This only reduces vLLM KV-cache
reservation and does not change the fixed NLL protocol.

Run with:

```bash
conda run -n wja-cospaq python \
  artifacts/debug/065_llama31_quality_calibration_size_sensitivity/scripts/run_all.py
```

No GPU is required for this final refitting step. Calibration cost combines the
real NLL elapsed time recorded by 057 and the extension labels, and is reported
in GPU-minutes.

The paper-facing `report/calibration_size_sensitivity.{png,pdf}` limits the
displayed calibration budget to 64 policies. The complete measured curve is
retained in `report/calibration_size_sensitivity_full.{png,pdf}`, and the CSV
and Markdown table retain all points through 72.
