# Canonical prefill-decode downstream-task validation

All listed Pareto policies were generated with the canonical sparse phase runtime. Solid ours markers use fresh-process measured speed under the common gpu_memory_utilization=0.80 protocol; hollow ours markers use only the roofline screening speed and are not final speed claims.

| policy | dataset | task score | measured ΔNLL | speed source | speedup |
|---|---|---:|---:|---|---:|
| point_001 | cnn_dm_1000 | 23.643 | 0.005825587757680051 | measured | 1.115 |
| point_001 | dsum | 21.316 | 0.005825587757680051 | measured | 1.115 |
| point_001 | IWSLT | 17.666 | 0.005825587757680051 | measured | 1.115 |
| point_002 | cnn_dm_1000 | 23.519 | 0.009304348305709365 | measured | 1.110 |
| point_002 | dsum | 21.449 | 0.009304348305709365 | measured | 1.110 |
| point_002 | IWSLT | 17.583 | 0.009304348305709365 | measured | 1.110 |
| point_003 | cnn_dm_1000 | 23.490 | 0.012211078099817252 | measured | 1.172 |
| point_003 | dsum | 21.302 | 0.012211078099817252 | measured | 1.172 |
| point_003 | IWSLT | 17.486 | 0.012211078099817252 | measured | 1.172 |
| point_004 | cnn_dm_1000 | 23.826 | 0.015245753880912538 | measured | 1.214 |
| point_004 | dsum | 21.740 | 0.015245753880912538 | measured | 1.214 |
| point_004 | IWSLT | 17.613 | 0.015245753880912538 | measured | 1.214 |
| point_005 | cnn_dm_1000 | 23.786 | 0.022388275291026938 | measured | 1.265 |
| point_005 | dsum | 21.262 | 0.022388275291026938 | measured | 1.265 |
| point_005 | IWSLT | 17.800 | 0.022388275291026938 | measured | 1.265 |
| point_006 | cnn_dm_1000 | 24.031 | 0.033364815967780403 | measured | 1.282 |
| point_006 | dsum | 20.300 | 0.033364815967780403 | measured | 1.282 |
| point_006 | IWSLT | 18.879 | 0.033364815967780403 | measured | 1.282 |
| point_008 | cnn_dm_1000 | 17.014 | 0.13122158282747987 | measured | 1.787 |
| point_008 | dsum | 14.002 | 0.13122158282747987 | measured | 1.787 |
| point_008 | IWSLT | 5.785 | 0.13122158282747987 | measured | 1.787 |
| point_009 | cnn_dm_1000 | 18.460 | 0.21203317674083944 | measured | 1.856 |
| point_009 | dsum | 13.574 | 0.21203317674083944 | measured | 1.856 |
| point_009 | IWSLT | 2.603 | 0.21203317674083944 | measured | 1.856 |
