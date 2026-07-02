# 064 FakeVLM Linear Time Proportion Plan

## Summary
- Add a FakeVLM dense BF16 linear time proportion study under `artifacts/debug/028_fakevlm_linear_time_proportion/`.
- Measure full FakeVLM prefill/decode latency and hook-based `nn.Linear` timing ratios.
- Keep existing FakeVLM 020/021/024/026 artifacts unchanged.

## Key Changes
- Add a benchmark script using `LlavaForConditionalGeneration` and FakeClue inputs.
- Default workloads:
  - `prefill_b1_i1024`: `batch_size=1,input_tokens=1024,output_tokens=0`
  - `prefill_b4_i1024`: `batch_size=4,input_tokens=1024,output_tokens=0`
  - `prefill_b16_i1024`: `batch_size=16,input_tokens=1024,output_tokens=0`
  - `prefill_b4_i4096`: `batch_size=4,input_tokens=4096,output_tokens=0`
  - `normal_01`: `batch_size=1,input_tokens=16384,output_tokens=32`
  - `normal_02`: `batch_size=1,input_tokens=16384,output_tokens=256`
- Hook every `nn.Linear` and report:
  - `all_linear`
  - `language_linear`
  - `vision_linear`
  - `projector_linear`
  - `other_linear`
- Add a 4-GPU task launcher with one workload per GPU at a time.
- Add summary CSV/Markdown report generation.

## Test Plan
- Static:
  - `python -m py_compile artifacts/debug/028_fakevlm_linear_time_proportion/scripts/run_linear_proportion.py`
  - `python -m py_compile artifacts/debug/028_fakevlm_linear_time_proportion/scripts/summarize.py`
  - `bash -n artifacts/debug/028_fakevlm_linear_time_proportion/scripts/launch_4gpu.sh`
- Runtime:
  - `bash artifacts/debug/028_fakevlm_linear_time_proportion/scripts/launch_4gpu.sh`
  - Confirm raw CSV has all workload rows or explicit OOM/error rows.
  - Confirm summary report identifies prefill/decode linear percentages and language/vision/projector split.

## Assumptions
- Target runtime uses local GPUs directly and conda env `cospaq`.
- Default FakeVLM path is `/home/agent/wja/data/models/lingcco/fakeVLM`.
- Default FakeClue test data paths match existing FakeVLM experiments.
- Dense BF16 only; no compressed backends are used in this study.
