# Llama2 phase-unified prefill quality recalibration

This isolated experiment rebuilds the 046 Llama2 72-policy fixed-WikiText NLL
dataset after removing two invalid baseline differences: legacy prepared-state
repacking and non-phase uniform runtime execution. All compressed policies use
phase-degenerate exports; the raw dense BF16 model remains the reference.

The proxy formula, frozen `p00`--`p53` fit split, `p54`--`p71` holdout split,
sample blocks, and local-error feature source are unchanged from 046.

Run in order:

```bash
conda activate cospaq
python artifacts/debug/053_llama2_prefill_phase_unified_quality_recalibration/scripts/bootstrap.py

conda activate vllm
python artifacts/debug/053_llama2_prefill_phase_unified_quality_recalibration/scripts/run_all.py --gpus 1,2,3,4,5,6,7

conda activate cospaq
python artifacts/debug/053_llama2_prefill_phase_unified_quality_recalibration/scripts/merge_nll.py
python artifacts/debug/053_llama2_prefill_phase_unified_quality_recalibration/scripts/fit_and_report.py
```
