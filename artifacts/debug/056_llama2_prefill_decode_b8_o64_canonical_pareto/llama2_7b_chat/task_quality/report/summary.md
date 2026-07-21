# Canonical prefill-decode downstream-task validation

All listed Pareto policies were generated with the canonical sparse phase runtime. Solid ours markers use fresh-process measured speed under the common gpu_memory_utilization=0.80 protocol; hollow ours markers use only the roofline screening speed and are not final speed claims.

| policy | dataset | task score | measured ΔNLL | speed source | speedup |
|---|---|---:|---:|---|---:|
| b8o64000 | cnn_dm_1000 | 23.610 |  | measured | 1.000 |
| b8o64000 | dsum | 21.598 |  | measured | 1.000 |
| b8o64000 | IWSLT | 19.412 |  | measured | 1.000 |
| b8o64003 | cnn_dm_1000 | 23.927 |  | measured | 1.040 |
| b8o64003 | dsum | 21.552 |  | measured | 1.040 |
| b8o64003 | IWSLT | 19.755 |  | measured | 1.040 |
| b8o64004 | cnn_dm_1000 | 23.798 |  | measured | 1.148 |
| b8o64004 | dsum | 21.401 |  | measured | 1.148 |
| b8o64004 | IWSLT | 21.083 |  | measured | 1.148 |
| b8o64005 | cnn_dm_1000 | 20.838 |  | measured | 1.189 |
| b8o64005 | dsum | 16.906 |  | measured | 1.189 |
| b8o64005 | IWSLT | 6.987 |  | measured | 1.189 |
| b8o64006 | cnn_dm_1000 | 19.266 |  | measured | 1.316 |
| b8o64006 | dsum | 15.867 |  | measured | 1.316 |
| b8o64006 | IWSLT | 7.152 |  | measured | 1.316 |
| b8o64007 | cnn_dm_1000 | 16.665 |  | measured | 1.442 |
| b8o64007 | dsum | 14.186 |  | measured | 1.442 |
| b8o64007 | IWSLT | 4.008 |  | measured | 1.442 |
| b8o64008 | cnn_dm_1000 | 17.940 |  | measured | 1.480 |
| b8o64008 | dsum | 13.218 |  | measured | 1.480 |
| b8o64008 | IWSLT | 2.790 |  | measured | 1.480 |
| b8o64009 | cnn_dm_1000 | 18.563 |  | measured | 1.561 |
| b8o64009 | dsum | 13.574 |  | measured | 1.561 |
| b8o64009 | IWSLT | 2.603 |  | measured | 1.561 |
