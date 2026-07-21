# Llama3.1 Dense-BF16 Prompt-Template Diagnosis Plan

## Goal

Determine whether the low absolute Llama3.1-8B-Instruct prefill-decode generation scores are substantially caused by the legacy/common prompt format, without modifying any main-experiment artifact.

## Fixed protocol

- Model: Llama3.1-8B-Instruct dense BF16 only.
- Backend: the existing vLLM PMPD generation backend.
- Samples, generation parameters, and length limits: identical between arms and drawn from the retained PMPD task protocol.
- Tasks: fixed small subsets of CNN/DM, DialogSum, and IWSLT (target 100 samples per task, or the protocol's available fixed prefix).
- Arms: existing legacy/common prompt; tokenizer `apply_chat_template(..., add_generation_prompt=True)` prompt.
- Stop behavior: explicitly record tokenizer EOS and Llama3 EOT ids; pass the same effective stop-token set to both arms.

## Steps and verification

1. Create isolated debug-062 scripts and manifest.  
   Verify: paths point only to dense Llama3 and fresh `debug/062` outputs.
2. Implement paired prompt construction and persistent vLLM evaluation.  
   Verify: prompt audit records identical source question ids, only template differs, and stop ids are recorded.
3. Run the two arms on identical fixed subsets.  
   Verify: each task/arm has the expected number of JSONL records and no duplicate ids.
4. Compute PMPD primary metrics, generation lengths, role-marker continuation counts, and representative paired samples.  
   Verify: report explicitly labels this as diagnosis only and does not write under 056/057/060 main artifacts.

## Out of scope

- No replacement of legacy/common main results.
- No uniform/ours re-evaluation, strategy solving, speed measurement, or changes to paper tables.
