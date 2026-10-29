# Phase Switch Decode Overhead Debug

Scenario: `input_len=2048, output_len=80, batch=16`.

The original decomposed benchmark showed phase-switch decode about 0.137s
slower than pure W4A16 Marlin. A same-session rerun with debug toggles shows
that this gap is mostly not the phase transition copy itself.

## Same-session comparison

| Method | Prefill | Decode | Total |
|---|---:|---:|---:|
| dense NVFP4 | 1.3581s | 2.9589s | 4.3170s |
| W4A16 Marlin | 2.2671s | 1.4576s | 3.7247s |
| phase baseline | 1.3799s | 1.5536s | 2.9335s |
| phase no trace | 1.4271s | 1.4365s | 2.8636s |
| phase no decode global GC | 1.4441s | 1.4964s | 2.9405s |

## Findings

- Phase baseline decode is 0.0959s slower than pure W4A16 in this rerun.
- Disabling per-Linear trace reduces phase decode by 0.1171s, or about 11.4us
  per decode Linear call.
- Disabling the decode-stage global release scan reduces phase decode by
  0.0572s, or about 5.6us per decode Linear call.
- The actual overlap boundary wait is only 0.0059s.
- The run has 10,240 decode Linear calls.

## Interpretation

The dominant measured gap is not the one-time phase transition. It comes from
artifact debug logic that runs on every decode Linear call:

- appending `apply_decode` trace events;
- scanning all phase-switch Linear layers in `_garbage_collect_ready()`;
- event/release state checks.

The prefill-stage release scan cannot simply be removed: doing so caused OOM,
because decode weights were copied to GPU while prefill weights were still
resident. The optimization target is therefore to keep prefill release
correctness while avoiding repeated decode-stage global scans and trace writes
in production.
