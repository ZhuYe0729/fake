# Llama-3.1-8B-Instruct: prefill-decode task validation

Protocol: B=8, input=2048, output=64; VLLM V1 `phase_hetero_mytest`; BF16 KV; chunked prefill disabled. All task scores use real phase-runtime generation. Speed is fresh-process measured E2E speed under the common 0.80 GPU-memory protocol.

`point_007` is retained as a measured data point but excluded from the line envelope because its speed measurement is anomalously below dense BF16 (0.739x).

Recommended paper candidates: `point_005` is the quality-preserving speed point (1.262x); `point_006` is the stronger-speed trade-off point (1.321x).

| policy | family | measured speedup | predicted ΔNLL | CNN R-L | CNN BERT | DialogSum R-L | DialogSum BERT | IWSLT BLEU |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dense BF16 | uniform | 1.000 |  | 19.389 | 83.582 | 13.482 | 78.278 | 10.777 |
| dense NVFP4 | uniform | 0.830 |  | 17.108 | 84.444 | 9.535 | 81.732 | 10.514 |
| sparse BF16 | uniform | 1.132 |  | 6.757 | 78.497 | 5.108 | 79.547 | 3.038 |
| sparse NVFP4 projection | uniform | 0.792 |  | 4.249 | 77.451 | 0.407 | 78.418 | 0.054 |
| W4A16/Marlin | uniform | 1.101 |  | 17.500 | 83.953 | 9.751 | 80.924 | 10.564 |
| point_000 | ours | 1.000 | 0.0000 | 19.389 | 83.582 | 13.482 | 78.278 | 10.777 |
| point_001 | ours | 1.030 | 0.0000 | 18.647 | 83.933 | 11.841 | 79.202 | 11.638 |
| point_002 | ours | 1.034 | 0.0018 | 18.587 | 83.938 | 11.249 | 79.440 | 10.945 |
| point_003 | ours | 1.077 | 0.0029 | 18.295 | 83.946 | 9.949 | 80.465 | 11.116 |
| point_004 | ours | 1.151 | 0.0082 | 18.689 | 83.866 | 10.923 | 80.218 | 10.764 |
| point_005 | ours | 1.262 | 0.0191 | 18.837 | 83.661 | 11.480 | 79.856 | 10.550 |
| point_006 | ours | 1.321 | 0.0394 | 16.779 | 84.176 | 9.305 | 81.209 | 10.312 |
| point_007 | ours | 0.739 | 0.0779 | 17.366 | 84.923 | 8.075 | 82.336 | 9.967 |
| point_008 | ours | 1.387 | 0.1302 | 8.218 | 80.184 | 6.372 | 80.759 | 5.587 |
| point_009 | ours | 1.390 | 0.2133 | 10.267 | 81.049 | 5.971 | 80.483 | 6.712 |
| point_010 | ours | 1.362 | 0.3407 | 14.623 | 83.016 | 4.616 | 79.874 | 0.910 |
| point_011 | ours | 1.420 | 0.3783 | 14.595 | 83.043 | 4.214 | 79.688 | 1.289 |
