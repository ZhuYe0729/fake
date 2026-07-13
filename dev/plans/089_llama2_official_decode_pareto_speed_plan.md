# Llama2-7B-Chat Decode Pareto Official Runner Realignment

## Goal

Replace the provisional decode Pareto speed curve measured with the `.8` / `benchmark_one.py` configuration by a curve measured under the previously established max-speed protocol, so that the max-speed endpoint retains its validated approximately 1.65x speedup.

## Evidence and assumptions

- The old max-speed checkpoint and Pareto decode point 11 have identical per-module phase assignments.
- The historical official runner uses scenario-specific `max_model_len=2128`, `max_num_seqs=16`, prefix cache disabled, phase-runtime environment flags, and `gpu_memory_utilization=0.9`.
- Reproducing point 11 with that runner gave 2974.8 ms, 1.636x relative to the historical 4868.1 ms dense-BF16 baseline.
- The provisional runner used `max_model_len=4096`, prefix cache enabled, and `.8`; its 1.40x endpoint must not be used as the formal result.
- Point 8 is currently infeasible under the formal `.9` configuration because its prefill NVFP4 workspace OOMs. It will be marked infeasible, not silently omitted.

## Steps

1. Implement a resumable official-runner speed harness and a separate output root.  
   Verify: it invokes `benchmark_phase_hetero.py` with the historical phase configuration and records output=1/80 runs separately from provisional results.
2. Test nearby intermediate policies for `.9` feasibility; select a feasible replacement for point 8 if needed.  
   Verify: successful vLLM run plus real WikiText NLL for any newly selected replacement.
3. Measure selected policies and all decode uniform references using the official runner.  
   Verify: repeated fresh-process runs on one GPU per task; all comparison rows share the same formal protocol.
4. Rebuild the decode Pareto CSV and speedup-quality figure.  
   Verify: max-speed endpoint is approximately 1.65x and every retained/excluded point has an explicit measured reason.
