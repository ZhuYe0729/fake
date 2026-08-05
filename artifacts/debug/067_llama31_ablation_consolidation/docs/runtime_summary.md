# vLLM phase-switch overlap ablation

Fixed protocol: Llama3.1-8B-Instruct, solved max-speed `point_011`, B=8,
S=2048, O=64, BF16 KV cache, chunked prefill disabled, and the common
`phase_hetero_mytest` backend. All 128 Linear modules change method between
prefill and decode. Formal timing disables tracing and alternates both variants
on the same GPU.

| Runtime variant | Median E2E latency (ms) ↓ | Samples |
|---|---:|---:|
| Synchronous phase-boundary copy | 1625.87 | 7 |
| Asynchronous overlapped prefetch | 1521.73 | 7 |

Asynchronous prefetch reduces median E2E latency by **6.41%**
(1.068× relative speedup) compared with exposing all decode-weight
copies at the phase boundary, and is faster in 5/7
paired comparisons. All samples are retained; this includes one visible async
long-tail sample. The experiment changes only
`PHASE_HETERO_SYNC_COPY`; it is a runtime ablation and does not alter the main
Pareto results.

## Direct overlap profile

| Mechanism metric | Median |
|---|---:|
| Total decode-state transfer work | 354.94 ms |
| Exposed copy stall | 1.92 ms |
| Hidden transfer work | 353.02 ms |
| Hidden ratio | 99.38% |

The mechanism values are medians of 3 independent
diagnostic runs using CUDA events. They are kept separate from formal E2E
timing because profiling adds instrumentation.


Paper-facing statement: *Overlapping decode-representation prefetch with the
remaining prefill computation reduces phase-switch latency by 6.4%
for a fully phase-heterogeneous policy. Direct CUDA-event profiling further shows that only 1.92 ms of 354.94 ms transfer work is exposed, hiding 99.4%.*
