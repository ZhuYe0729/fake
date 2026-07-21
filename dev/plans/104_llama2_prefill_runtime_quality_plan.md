# Llama2 prefill-only runtime-faithful quality plan

## Goal

Replace the invalid Transformers weight-proxy quality values for the
Llama2-7B-Chat prefill-only experiment with values obtained from the actual
vLLM quantized runtime.  The scope is all uniform methods and all currently
reported `ours_point_*` policies.  Llama-3.1 and prefill-decode are explicit
TODOs for a later plan.

## Scope and assumptions

- Existing vLLM speed measurements remain authoritative and are not rerun.
- Quality tasks are WikiText PPL, Winogrande, ARC-Easy, MMLU, plus the existing
  ARC-Challenge closure where practical.
- Every policy is evaluated with its real vLLM checkpoint/quant method:
  uniform checkpoints use their native quantization config; ours uses the
  phase-heterogeneous runtime and policy materialization path.
- The current `041` HFLM results are retained only as an invalid weight-proxy
  debug artifact and are never merged into paper-facing tables.

## Steps

1. Build a small vLLM likelihood adapter around prompt logprobs for rolling
   WikiText NLL/PPL and multiple-choice option likelihood.  Preserve the
   existing lm-eval prompt/few-shot conventions where possible.
   - Verify: dense BF16 runs end-to-end; dense NVFP4 and Marlin NVFP4 produce
     distinguishable logits on the same fixed prompts.
2. Add a runtime checkpoint resolver for all Llama2 uniform methods and all
   published ours policies, including phase-heterogeneous policy activation.
   - Verify: resolver records the actual model path, quant method, policy, and
     vLLM runtime configuration for each run.
3. Run a representative runtime audit: dense BF16, dense NVFP4, Marlin NVFP4,
   sparse BF16, sparse NVFP4, and one mixed ours policy.
   - Verify: dense NVFP4/Marlin no longer collapse to bit-identical scores;
     each result has a real vLLM provenance record.
4. Run all Llama2 prefill-only uniform and ours policies on the selected tasks,
   using GPU 1-7 and resumable per-policy/per-task outputs.
   - Verify: complete matrix with no proxy backend; per-result logs capture
     prompt-logprob settings and runtime quantization config.
5. Produce a new debug-only joined table/plots with existing measured speeds.
   Mark `041` superseded for Llama2 runtime-quality comparisons.
   - Verify: only runtime-faithful results feed the new table.

## TODO outside this plan

- Apply the same runtime evaluator to Llama-3.1-8B-Instruct prefill-only.
- Replace prefill-decode proxy WikiText NLL with actual vLLM likelihood NLL.
- Leave already-real vLLM prefill-decode generation scores and all speed
  results unchanged.
