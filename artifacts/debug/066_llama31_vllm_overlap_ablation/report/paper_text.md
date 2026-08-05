### Runtime overlap ablation

We further ablate the asynchronous state-prefetch mechanism used at the
prefill-to-decode transition. We compare it with a synchronous baseline that
exposes all host-to-device transfers at the phase boundary. Both variants use
the same Llama3.1-8B-Instruct max-speed policy, execution backend, checkpoint,
and B=8/S=2048/O=64 workload; the policy switches the execution method of all
128 Linear modules. We alternate the two variants on the same GPU and report
the median of seven runs with tracing disabled. Asynchronous prefetch reduces
E2E latency from 1625.87 ms to 1521.73 ms, corresponding to a 6.41% latency
reduction (1.068x speedup). This result confirms that prefetching decode
representations while later prefill layers are still executing hides a
meaningful portion of the otherwise exposed phase-transition overhead. Direct
CUDA-event profiling supports this interpretation: across three independent
runs, the median decode-state transfer work is 354.94 ms, while only 1.92 ms
is exposed as copy-induced compute-stream stalls. Thus, the asynchronous
schedule hides 99.4% of the transfer work.
