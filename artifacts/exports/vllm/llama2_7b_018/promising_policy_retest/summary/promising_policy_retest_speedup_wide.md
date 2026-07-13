# Promising Scenario Speedup Wide Table

All speedups are measured vLLM median-latency speedups over dense bf16 for the same `(batch, input_seq, output_seq)` scenario. `*_acc_norm` columns are full ARC-Challenge 0-shot `acc_norm` measured with vLLM + lm-eval.

| batch | input_seq | output_seq | dense_bf16 | dense_nvfp4 | sparse_bf16 | sparse_nvfp4 | marlin_nvfp4 | original_hetero | optimized_hetero | opt_vs_best_single | optimized_acc_norm | optimized_policy | max_speed_hetero | max_speed_vs_best_single | max_speed_acc_norm | max_speed_policy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| 2 | 16384 | 128 | 1.000 | 6.889 | 8.074 | 6.679 | 8.530 | 8.527 | 7.255 | 0.851 (marlin_nvfp4) | 0.4428 | policy_000_4f5ae62f3f | 7.371 | 0.864 (marlin_nvfp4) | 0.4428 | maxspeed_000_4f5ae62f3f |
| 2 | 16384 | 64 | 1.000 | 5.669 | 6.324 | 5.615 | 5.883 | 6.350 | 5.653 | 0.894 (sparse_bf16) | 0.4309 | policy_001_2ac520a243 | 5.708 | 0.903 (sparse_bf16) | 0.4309 | maxspeed_001_2ac520a243 |
| 4 | 16384 | 128 | 1.000 | 3.867 | 3.751 | 4.098 | 2.716 | 3.360 | 3.695 | 0.902 (sparse_nvfp4) | 0.4428 | policy_000_4f5ae62f3f | 3.694 | 0.901 (sparse_nvfp4) | 0.4428 | maxspeed_000_4f5ae62f3f |
| 4 | 16384 | 64 | 1.000 | 3.648 | 3.519 | 3.870 | 2.545 | 3.150 | 3.664 | 0.947 (sparse_nvfp4) | 0.4309 | policy_001_2ac520a243 | 3.599 | 0.930 (sparse_nvfp4) | 0.4309 | maxspeed_001_2ac520a243 |
| 8 | 16384 | 128 | 1.000 | 3.359 | 3.212 | 3.605 | 2.244 | 2.824 | 12.798 | 3.550 (sparse_nvfp4) | 0.3618 | policy_002_605d24248e | 3.195 | 0.886 (sparse_nvfp4) | 0.3362 | maxspeed_002_1556299ab3 |
| 8 | 16384 | 64 | 1.000 | 3.226 | 3.089 | 3.473 | 2.158 | 2.717 | 9.643 | 2.777 (sparse_nvfp4) | 0.3660 | policy_003_23b5bafdf0 | 3.069 | 0.884 (sparse_nvfp4) | 0.3362 | maxspeed_002_1556299ab3 |
| 4 | 1024 | 1 | 1.000 | 2.126 | 1.743 | 2.470 | 1.098 | 1.858 | 2.021 | 0.818 (sparse_nvfp4) | 0.4309 | policy_001_2ac520a243 | 2.179 | 0.882 (sparse_nvfp4) | 0.4317 | maxspeed_003_74a8a502ad |
| 2 | 1024 | 1 | 1.000 | 2.118 | 1.729 | 2.455 | 1.190 | 1.945 | 2.166 | 0.882 (sparse_nvfp4) | 0.4309 | policy_001_2ac520a243 | 2.270 | 0.925 (sparse_nvfp4) | 0.4317 | maxspeed_003_74a8a502ad |
| 4 | 512 | 1 | 1.000 | 2.058 | 1.723 | 2.286 | 1.178 | 1.819 | 1.937 | 0.847 (sparse_nvfp4) | 0.4309 | policy_001_2ac520a243 | 2.036 | 0.891 (sparse_nvfp4) | 0.4317 | maxspeed_003_74a8a502ad |
| 2 | 4096 | 1 | 1.000 | 2.010 | 1.676 | 2.259 | 1.071 | 1.818 | 1.875 | 0.830 (sparse_nvfp4) | 0.4309 | policy_001_2ac520a243 | 2.029 | 0.898 (sparse_nvfp4) | 0.4317 | maxspeed_003_74a8a502ad |
| 1 | 4096 | 1 | 1.000 | 2.008 | 1.683 | 2.251 | 1.072 | 1.796 | 1.858 | 0.825 (sparse_nvfp4) | 0.4309 | policy_001_2ac520a243 | 2.014 | 0.894 (sparse_nvfp4) | 0.4317 | maxspeed_003_74a8a502ad |
| 8 | 512 | 1 | 1.000 | 1.945 | 1.669 | 2.187 | 1.052 | 1.409 | 1.934 | 0.884 (sparse_nvfp4) | 0.4369 | policy_004_4831f76351 | 2.229 | 1.020 (sparse_nvfp4) | 0.4087 | maxspeed_004_f2600ffcfc |
| 128 | 256 | 64 | 1.000 | 1.535 | 1.776 | 1.579 | 1.469 | 1.719 | 1.443 | 0.812 (sparse_bf16) | 0.4061 | policy_005_023ac1246d | 1.934 | 1.089 (sparse_bf16) | 0.2884 | maxspeed_005_4746310a30 |
| 32 | 4096 | 16 | 1.000 | 1.645 | 1.525 | 1.734 | 1.084 | 1.366 | 1.357 | 0.783 (sparse_nvfp4) | 0.3660 | policy_006_4db717c6ec | 1.529 | 0.882 (sparse_nvfp4) | 0.3362 | maxspeed_002_1556299ab3 |
| 256 | 512 | 16 | 1.000 | 1.691 | 1.563 | 1.732 | 1.082 | 1.388 | 1.394 | 0.805 (sparse_nvfp4) | 0.3660 | policy_006_4db717c6ec | 1.570 | 0.906 (sparse_nvfp4) | 0.3362 | maxspeed_002_1556299ab3 |
| 64 | 512 | 16 | 1.000 | 1.619 | 1.636 | 1.713 | 1.139 | 1.453 | 1.717 | 1.002 (sparse_nvfp4) | 0.4266 | policy_007_f74a83b67c | 1.959 | 1.143 (sparse_nvfp4) | 0.2884 | maxspeed_005_4746310a30 |

Notes:

- `original_hetero` is the previously measured broad-grid hetero baseline.
- `optimized_hetero` is the newly exported and measured P024 quality-budget layer-wise heterogeneous policy for each scenario.
- `opt_vs_best_single` is `optimized_hetero` latency speedup over the fastest measured single method in that scenario; the method is shown in parentheses.
- `max_speed_hetero` is the newly exported and measured unconstrained speed-optimal heterogeneous policy for each scenario.

## Single-Method Quality

| method | ARC-C acc | ARC-C acc_norm | NLL | sample_len | source |
|---|---:|---:|---:|---:|---|
| dense_bf16 | 0.4292 | 0.4514 | 2.0395 | 1172 | 018_uniform_full_arc_c |
| dense_nvfp4 | 0.4164 | 0.4377 | 2.1215 | 1172 | 018_uniform_full_arc_c |
| sparse_bf16 | 0.2969 | 0.3379 | 2.3898 | 1172 | 018_uniform_full_arc_c |
| sparse_nvfp4 | 0.1843 | 0.2287 | 3.3579 | 1172 | 018_uniform_full_arc_c |
| marlin_nvfp4 | 0.4283 | 0.4360 | 2.0942 | 1172 | 018_uniform_full_arc_c |

## Optimized Policy Details

| kind | policy | ARC-C acc_norm | quality_cost | method_counts | scenarios | assignment_summary |
|---|---|---:|---:|---|---|---|
| optimized_hetero | policy_000_4f5ae62f3f | 0.4428 | 0.216190828861 | dense=64, dnvfp4=64, sbf16=0, snvfp4=0 | b2_in16384_out128, b4_in16384_out128 | dense_bf16: mlp.down_proj@0-31; self_attn.o_proj@0-31 <br> dense_nvfp4: mlp.gate_up_proj@0-31; self_attn.qkv_proj@0-31 |
| optimized_hetero | policy_001_2ac520a243 | 0.4309 | 0.266931300353 | dense=32, dnvfp4=96, sbf16=0, snvfp4=0 | b2_in16384_out64, b4_in16384_out64, b4_in1024_out1, b2_in1024_out1, b4_in512_out1, b2_in4096_out1, b1_in4096_out1 | dense_bf16: self_attn.o_proj@0-31 <br> dense_nvfp4: mlp.down_proj@0-31; mlp.gate_up_proj@0-31; self_attn.qkv_proj@0-31 |
| optimized_hetero | policy_002_605d24248e | 0.3618 | 0.271049145648 | dense=54, dnvfp4=1, sbf16=73, snvfp4=0 | b8_in16384_out128 | dense_bf16: mlp.down_proj@5; self_attn.o_proj@1-31; self_attn.qkv_proj@7,9,11-14,16-31 <br> dense_nvfp4: mlp.gate_up_proj@1 <br> sparse_bf16: mlp.down_proj@0-4,6-31; mlp.gate_up_proj@0,2-31; self_attn.o_proj@0; self_attn.qkv_proj@0-6,8,10,15 |
| optimized_hetero | policy_003_23b5bafdf0 | 0.3660 | 0.270692073268 | dense=53, dnvfp4=1, sbf16=74, snvfp4=0 | b8_in16384_out64 | dense_bf16: self_attn.o_proj@1-30; self_attn.qkv_proj@7,9-14,16-31 <br> dense_nvfp4: mlp.gate_up_proj@1 <br> sparse_bf16: mlp.down_proj@0-31; mlp.gate_up_proj@0,2-31; self_attn.o_proj@0,31; self_attn.qkv_proj@0-6,8,15 |
| optimized_hetero | policy_004_4831f76351 | 0.4369 | 0.267393081971 | dense=30, dnvfp4=83, sbf16=15, snvfp4=0 | b8_in512_out1 | dense_bf16: self_attn.o_proj@1-24,27-31; self_attn.qkv_proj@31 <br> dense_nvfp4: mlp.down_proj@2-14,17,19-22,24-25,27; mlp.gate_up_proj@0-31; self_attn.qkv_proj@1-30 <br> sparse_bf16: mlp.down_proj@0-1,15-16,18,23,26,28-31; self_attn.o_proj@0,25-26; self_attn.qkv_proj@0 |
| optimized_hetero | policy_005_023ac1246d | 0.4061 | 0.277551275605 | dense=62, dnvfp4=12, sbf16=54, snvfp4=0 | b128_in256_out64 | dense_bf16: mlp.down_proj@2-17,19-22,24-25,27-29; self_attn.o_proj@1-31; self_attn.qkv_proj@19,24-25,29-31 <br> dense_nvfp4: mlp.gate_up_proj@0-3,5-8,19-20,24; self_attn.qkv_proj@20 <br> sparse_bf16: mlp.down_proj@0-1,18,23,26,30-31; mlp.gate_up_proj@4,9-18,21-23,25-31; self_attn.o_proj@0; self_attn.qkv_proj@0-18,21-23,26-28 |
| optimized_hetero | policy_006_4db717c6ec | 0.3660 | 0.270673371768 | dense=53, dnvfp4=2, sbf16=73, snvfp4=0 | b32_in4096_out16, b256_in512_out16 | dense_bf16: self_attn.o_proj@1-31; self_attn.qkv_proj@7,9,11-14,16-31 <br> dense_nvfp4: mlp.gate_up_proj@0-1 <br> sparse_bf16: mlp.down_proj@0-31; mlp.gate_up_proj@2-31; self_attn.o_proj@0; self_attn.qkv_proj@0-6,8,10,15 |
| optimized_hetero | policy_007_f74a83b67c | 0.4266 | 0.267033012553 | dense=34, dnvfp4=64, sbf16=30, snvfp4=0 | b64_in512_out16 | dense_bf16: self_attn.o_proj@1-30; self_attn.qkv_proj@21,29-31 <br> dense_nvfp4: mlp.down_proj@2,5,7-9,19-20,24; mlp.gate_up_proj@0-31; self_attn.qkv_proj@3,5-20,22-28 <br> sparse_bf16: mlp.down_proj@0-1,3-4,6,10-18,21-23,25-31; self_attn.o_proj@0,31; self_attn.qkv_proj@0-2,4 |
| max_speed_hetero | maxspeed_000_4f5ae62f3f | 0.4428 | 0.216190828861 | dense=64, dnvfp4=64, sbf16=0, snvfp4=0 | b2_in16384_out128, b4_in16384_out128 | dense_bf16: mlp.down_proj@0-31; self_attn.o_proj@0-31 <br> dense_nvfp4: mlp.gate_up_proj@0-31; self_attn.qkv_proj@0-31 |
| max_speed_hetero | maxspeed_001_2ac520a243 | 0.4309 | 0.266931300353 | dense=32, dnvfp4=96, sbf16=0, snvfp4=0 | b2_in16384_out64, b4_in16384_out64 | dense_bf16: self_attn.o_proj@0-31 <br> dense_nvfp4: mlp.down_proj@0-31; mlp.gate_up_proj@0-31; self_attn.qkv_proj@0-31 |
| max_speed_hetero | maxspeed_002_1556299ab3 | 0.3362 | 0.530133636441 | dense=0, dnvfp4=0, sbf16=128, snvfp4=0 | b8_in16384_out128, b8_in16384_out64, b32_in4096_out16, b256_in512_out16 | sparse_bf16: mlp.down_proj@0-31; mlp.gate_up_proj@0-31; self_attn.o_proj@0-31; self_attn.qkv_proj@0-31 |
| max_speed_hetero | maxspeed_003_74a8a502ad | 0.4317 | 0.32132498665 | dense=0, dnvfp4=128, sbf16=0, snvfp4=0 | b4_in1024_out1, b2_in1024_out1, b4_in512_out1, b2_in4096_out1, b1_in4096_out1 | dense_nvfp4: mlp.down_proj@0-31; mlp.gate_up_proj@0-31; self_attn.o_proj@0-31; self_attn.qkv_proj@0-31 |
| max_speed_hetero | maxspeed_004_f2600ffcfc | 0.4087 | 0.378773941307 | dense=0, dnvfp4=64, sbf16=64, snvfp4=0 | b8_in512_out1 | dense_nvfp4: mlp.gate_up_proj@0-31; self_attn.qkv_proj@0-31 <br> sparse_bf16: mlp.down_proj@0-31; self_attn.o_proj@0-31 |
| max_speed_hetero | maxspeed_005_4746310a30 | 0.2884 | 0.81435225446 | dense=0, dnvfp4=0, sbf16=96, snvfp4=32 | b128_in256_out64, b64_in512_out16 | sparse_bf16: mlp.down_proj@0-31; self_attn.o_proj@0-31; self_attn.qkv_proj@0-31 <br> sparse_nvfp4: mlp.gate_up_proj@0-31 |
