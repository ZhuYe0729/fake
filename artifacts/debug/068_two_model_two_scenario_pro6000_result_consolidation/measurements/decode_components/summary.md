# Pro 6000 decode component timing

RTX 5090-compatible protocol: fresh vLLM process/model per sample; O=1 and O=64 each use 1 warmup + 5 measured runs; TPOT=(median E2E-median TTFT)/63.

| Model | Method | Policy | TTFT ms | TPOT ms | E2E ms | TTFT speedup | TPOT speedup | E2E speedup |
|---|---|---|---:|---:|---:|---:|---:|---:|
| llama2_7b_chat | BF16 | uniform_p00 | 685.251 | 18.298 | 1838.011 | 1.000× | 1.000× | 1.000× |
| llama2_7b_chat | Dense NVFP4 | uniform_p01 | 637.528 | 25.853 | 2266.251 | 1.075× | 0.708× | 0.811× |
| llama2_7b_chat | Sparse BF16 | uniform_p02 | 524.765 | 16.910 | 1590.117 | 1.306× | 1.082× | 1.156× |
| llama2_7b_chat | Sparse NVFP4 | uniform_p03 | 581.160 | 26.960 | 2279.668 | 1.179× | 0.679× | 0.806× |
| llama2_7b_chat | W4A16 Marlin | uniform_p04 | 757.431 | 12.000 | 1513.449 | 0.905× | 1.525× | 1.214× |
| llama2_7b_chat | Ours (Max speed) | point_018 | 515.974 | 11.988 | 1271.195 | 1.328× | 1.526× | 1.446× |
| llama2_7b_chat | Ours (Balanced) | point_014 | 592.607 | 12.734 | 1394.829 | 1.156× | 1.437× | 1.318× |
| llama31_8b_instruct | BF16 | uniform_p00 | 694.497 | 15.473 | 1669.327 | 1.000× | 1.000× | 1.000× |
| llama31_8b_instruct | Dense NVFP4 | uniform_p01 | 653.193 | 24.330 | 2186.011 | 1.063× | 0.636× | 0.764× |
| llama31_8b_instruct | Sparse BF16 | uniform_p02 | 535.574 | 14.867 | 1472.202 | 1.297× | 1.041× | 1.134× |
| llama31_8b_instruct | Sparse NVFP4 | uniform_p03 | 591.320 | 27.340 | 2313.766 | 1.174× | 0.566× | 0.721× |
| llama31_8b_instruct | W4A16 Marlin | uniform_p04 | 766.517 | 8.747 | 1317.606 | 0.906× | 1.769× | 1.267× |
| llama31_8b_instruct | Ours (Max speed) | point_020 | 488.461 | 8.756 | 1040.066 | 1.422× | 1.767× | 1.605× |
| llama31_8b_instruct | Ours (Balanced) | point_013 | 587.081 | 10.438 | 1244.647 | 1.183× | 1.482× | 1.341× |
