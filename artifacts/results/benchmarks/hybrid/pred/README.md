# Predictor Hybrid vs Manual Hybrid

## Layout

Results are split by scenario, matching the `manual/` directory:

- `prefill_only/`: prefill-only policies, strategy diffs, module timings, and `prefill_only_predictor_hybrid_summary.md`
- `normal_01/`: prefill+decode policies, strategy diffs, module timings, full E2E, and `predictor_hybrid_e2e_summary.md`
- `Predictor_Hybrid_Analysis.md`: cross-scenario comparison and interpretation

## Scenarios
- `prefill_only`: batch_size=16, input_tokens=1024, output_tokens=0.
- `normal_01`: batch_size=1, input_tokens=16384, output_tokens=32.

Predicted latency is summed over compressible Linear layers using `KernelLatencyPredictor`; manual columns come from existing benchmark artifacts.

## Predictor Hybrid Summary

| Model | Scenario | Pred hybrid linear ms | Pred speedup | Manual hybrid ms | Manual speedup |
|---|---|---:|---:|---:|---:|
| Llama-2-7B | prefill_only | 434.4850 | 2.2663x | 413.9049 | 2.1945x |
| Llama-2-7B | normal_01 | 989.5526 | 1.3983x | 1930.0000 | 1.2600x |
| Llama-3.1-8B | prefill_only | 433.0726 | 2.4886x | 405.3724 | 2.4285x |
| Llama-3.1-8B | normal_01 | 937.9414 | 1.6028x | 2002.0000 | 1.1300x |
| Qwen3.5-9B | prefill_only | 442.2445 | 2.4052x | 427.2405 | 2.2766x |
| Qwen3.5-9B | normal_01 | 964.4750 | 1.5598x | 3308.0000 | 1.2700x |

## GPU Module E2E Check

This table uses the predictor-selected policy, builds the selected kernel module for each linear group on GPU, times prefill/decode, and sums over layer counts. It is a real kernel/module timing check, not a full model forward.

| Model | Scenario | GPU policy prefill ms | GPU policy decode×n ms | GPU policy linear E2E ms | Manual hybrid ms |
|---|---|---:|---:|---:|---:|
| Llama-2-7B | prefill_only | 418.6518 | 0.0000 | 418.6518 | 413.9049 |
| Llama-2-7B | normal_01 | 597.9267 | 266.7430 | 864.6696 | 1930.0000 |
| Llama-3.1-8B | prefill_only | 409.1248 | 0.0000 | 409.1248 | 405.3724 |
| Llama-3.1-8B | normal_01 | 592.1401 | 353.5847 | 945.7249 | 2002.0000 |
| Qwen3.5-9B | prefill_only | 426.1789 | 0.0000 | 426.1789 | 427.2405 |
| Qwen3.5-9B | normal_01 | 584.4934 | 321.7007 | 906.1940 | 3308.0000 |

## Full Model E2E

This table loads the full Hugging Face model, applies the predictor policy to all selected Linear layers, and runs the actual model forward. `normal_01` is directly comparable to the existing manual full E2E summary. The existing `prefill_only` manual artifact is a linear/module-level benchmark, so it is shown only as a reference.

| Model | Scenario | Replaced | Skipped | Predictor policy | Full prefill ms | Full decode×n ms | Full E2E ms | Manual reference ms |
|---|---|---:|---:|---|---:|---:|---:|---:|
| Llama-2-7B | normal_01 | 224 | 0 | `dense_nvfp4/marlin_nvfp4:224` | 1183.5671 | 1070.8083 | 2254.3754 | 1930.0000 |
| Llama-3.1-8B | normal_01 | 224 | 0 | `dense_nvfp4/marlin_nvfp4:160,bf16:64` | 1206.1615 | 954.6636 | 2160.8251 | 2002.0000 |
| Qwen3.5-9B | normal_01 | 248 | 0 | `dense_nvfp4/marlin_nvfp4:184,bf16:64` | 2510.0425 | 1694.7151 | 4204.7576 | 3308.0000 |
| Llama-2-7B | prefill_only | 224 | 0 | `sparse_bf16:160,sparse_nvfp4:64` | 1172.6062 | 0.0000 | 1172.6062 | 413.9049 |
| Llama-3.1-8B | prefill_only | 224 | 0 | `sparse_bf16:160,sparse_nvfp4:64` | 1215.0254 | 0.0000 | 1215.0254 | 405.3724 |

## Strategy Difference Summary

| Model | Scenario | Manual | Predictor | Groups | Layers |
|---|---|---|---|---:|---:|
| Llama-2-7B | prefill_only | sparse_bf16->sparse_bf16 | sparse_bf16->sparse_bf16 | 5 | 160 |
| Llama-2-7B | prefill_only | sparse_nvfp4->sparse_nvfp4 | sparse_nvfp4->sparse_nvfp4 | 2 | 64 |
| Llama-2-7B | normal_01 | dense_nvfp4->marlin_nvfp4 | dense_nvfp4->marlin_nvfp4 | 7 | 224 |
| Llama-3.1-8B | prefill_only | sparse_bf16->sparse_bf16 | sparse_bf16->sparse_bf16 | 5 | 160 |
| Llama-3.1-8B | prefill_only | sparse_nvfp4->sparse_nvfp4 | sparse_nvfp4->sparse_nvfp4 | 2 | 64 |
| Llama-3.1-8B | normal_01 | dense_nvfp4->marlin_nvfp4 | dense_bf16->dense_bf16 | 2 | 64 |
| Llama-3.1-8B | normal_01 | dense_nvfp4->marlin_nvfp4 | dense_nvfp4->marlin_nvfp4 | 5 | 160 |
| Qwen3.5-9B | prefill_only | marlin_nvfp4->marlin_nvfp4 | sparse_bf16->sparse_bf16 | 2 | 16 |
| Qwen3.5-9B | prefill_only | sparse_bf16->sparse_bf16 | sparse_bf16->sparse_bf16 | 6 | 136 |
| Qwen3.5-9B | prefill_only | sparse_nvfp4->sparse_nvfp4 | sparse_nvfp4->sparse_nvfp4 | 4 | 96 |
| Qwen3.5-9B | normal_01 | dense_bf16->dense_bf16 | dense_bf16->dense_bf16 | 4 | 64 |
| Qwen3.5-9B | normal_01 | sparse_bf16->sparse_bf16 | dense_nvfp4->marlin_nvfp4 | 8 | 184 |
