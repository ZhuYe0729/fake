# Llama3.1 prefill-only vLLM runtime quality plan

Run the same real vLLM likelihood closure as Llama2 for five uniform and seven
published ours prefill-only policies. Reuse uniform checkpoints, materialize
missing phase-heterogeneous ours checkpoints under debug 043, then evaluate
WikiText, Winogrande, ARC-Easy, ARC-Challenge, and MMLU without remeasuring
speed.

1. Build and audit the exact runtime policy inventory → verify that each NVFP4
   method uses its distinct vLLM quantization path.
2. Materialize and verify the seven missing phase-heterogeneous checkpoints →
   verify their exported policy matches the selected solver JSON.
3. Evaluate all five tasks with vLLM and join them with measured speed → verify
   every reported row has five actual quality results.
