# External backend contract

## Patched vLLM

Repository and audited base revision:

```text
https://github.com/ZhuYe0729/vllm-cospaq.git
679527f6fffdc51ee79a92b4d9b0749a8855e5b6
```

DASO expects a vLLM 0.11.1-compatible tree containing:

- `vllm/` importable source;
- `artifacts/dev/012_phase_hetero_linear/export_phase_hetero_model.py`;
- the `phase_hetero_mytest` quantization method;
- canonical sparse BF16/NVFP4 import without runtime `--prune`;
- prefill/decode method switching and asynchronous materialization controls;
- PMPD evaluation under `artifacts/dev/011_phase_switch_linear_test`.

All uniform and heterogeneous policies must be exported through this same
runtime. Native uniform runners must not be mixed into a comparison.

The final experiment machine also contained uncommitted changes to the
following files relative to that revision:

```text
artifacts/dev/011_phase_switch_linear_test/pmpd_vllm_eval.py
artifacts/dev/012_phase_hetero_linear/export_phase_hetero_model.py
vllm/vllm/engine/arg_utils.py
```

The migration bundle records these changes as an explicit release patch and
source snapshots in
[`artifacts/rebuttal/portability/dependencies/phase_vllm`](../../artifacts/rebuttal/portability/dependencies/phase_vllm).
See the [PRO 6000 handoff](../../artifacts/rebuttal/PRO6000_HANDOFF.md) for the
locked base and application commands. Rebuild on the target machine and verify
the numerical environment before claiming GPU reproduction; this package does
not silently replace an external working tree.

## CUTLASS wrapper

Repository and audited revision:

```text
https://github.com/ZhuYe0729/cutlass_wrapper.git
8f60a9d2251e39fc7adbb385f6612260db56ec38
```

Follow the wrapper repository's `INSTALL.md` to compile the CUDA extensions
against the same CUDA, PyTorch, and GPU architecture used by vLLM.

The wrapper must expose the five runtime modules and the calibrated predictor:

```text
NVFP4Linear
SparseBF16Linear
SparseNVFP4Linear
MarlinNVFP4Linear
modeling/kernel_predictor.py:KernelLatencyPredictor
```

Set paths explicitly:

```bash
export DASO_VLLM_ROOT=/path/to/patched/vllm
export DASO_CUTLASS_WRAPPER_ROOT=/path/to/cutlass_wrapper
```

Changing GPU, CUDA, PyTorch, vLLM, or wrapper ABI requires rebuilding the
extensions and re-profiling latency. Quality calibration can be reused only
when the numerical runtime and canonical states remain identical.

The paper evaluation environment used `lm-eval==0.4.12`; the DASO `eval`
optional dependency pins this revision because metric names and vLLM adapter
behavior are part of the measurement protocol.
