# Real-vLLM prefill-only quality recalibration

This isolated bundle replaces the old Transformers/prepared-weight NLL label
with direct vLLM prompt-logprob NLL.  It deliberately keeps the existing
local-error feature tables and the positive local+global quality-model form.

## Protocol

- Models: Llama2-7B-Chat and Llama3.1-8B-Instruct.
- Labels: mean token NLL over the same 100 fixed WikiText blocks of 2048
  scored tokens per model.
- Policies: 72 deterministic prefill-only policies, with `p00`--`p53` for
  fitting and `p54`--`p71` as frozen holdout.
- Uniform `p00`--`p04` use the established uniform checkpoints.  `p05+` are
  temporary phase-heterogeneous exports and are removed after scoring.

## Commands

Run the generation/fit/report steps from the repository root in `cospaq`.
Run `run_all.py` in the `vllm` environment, which contains the vLLM runtime
dependencies used by the established evaluation scripts:

```bash
conda activate cospaq
python artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/scripts/generate_inputs.py --model llama2
conda activate vllm
python artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/scripts/run_all.py --model llama2 --gpus 1
conda activate cospaq
python artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/scripts/merge_nll.py --model llama2
python artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/scripts/fit_quality_proxy.py --model llama2
python artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/scripts/build_report.py --model llama2
```

Use `--selection p00,p01` for a semantic smoke test.  `run_all.py` is
restartable: a result with the expected policy/sample hashes is not rerun.

The same commands support `--model llama31`.  Results are intentionally not
promoted to `artifacts/exports/` by this bundle.
