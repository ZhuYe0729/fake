# Llama2 prefill-only real vLLM quality

This experiment supersedes the Llama2 portion of `041` for runtime quality
comparison.  It uses lm-eval's vLLM adapter and actual checkpoint quantization,
including activation quantization, rather than injecting proxy BF16 weights into
Transformers.

It evaluates all five uniform methods and all nine published ours policies on
WikiText, Winogrande, ARC-Easy, ARC-Challenge, and MMLU (0-shot).  Existing
prefill-only vLLM speed values are read only and are not remeasured.

```bash
conda run -n vllm python scripts/build_manifest.py
conda run -n vllm python scripts/run_all.py --selection dense_bf16,dense_nvfp4,marlin_nvfp4,sparse_bf16,sparse_nvfp4,ours_point_012 --audit
conda run -n vllm python scripts/run_all.py --gpus 1,2,3,4,5,6,7
conda run -n vllm python scripts/summarize.py
```

All results remain below this directory.  `041` is a weight-proxy artifact and
must not be used for Llama2 runtime-quality claims.
