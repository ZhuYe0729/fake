# 001 — Qwen3-8B / HAWQ-v3 allocation reproduction

Status: complete, validated on 2026-10-25. Start with [REPORT.md](REPORT.md) and
[results/summary_final.csv](results/summary_final.csv). All 39 policy IDs have
sequential physical-GPU-3 speed measurements; 46 scenario/NLL points, 65 main
likelihood-task results, and 45 full generation results are complete.
`results/final_audit.json` validates 42,495 generated examples, 144 calibration
labels, frozen input hashes and command/code snapshots. The two prototype W4A4
task files are retained as diagnostics and excluded from the 65 main results.

Scope: the original five uniform baselines, INT8, Marlin INT4, adapted HAWQ-v3
Hessian/ILP allocation, DASO, and a phase-uniform hybrid on Qwen3-8B.
The frozen settings are in `configs/final_protocol.json`. Main INT4 uses
W4A16 with native BF16 activations and group size 128. Main INT8 uses genuine
integer GEMM (`torch._int_mm`), with per-token activation and per-output weight
scaling; the installed vLLM CUTLASS INT8 kernel rejects this SM120 GPU.
All actual packaged Linear timings include quantization, padding and scaling.

HAWQ-v3 is an **adapted allocator**, not a reproduction of the original CNN
QAT/integer-only TVM pipeline. `reference/sources.json` pins the official source.
Its objective is the sum of normalized true NLL Hessian traces times squared
weight perturbations, under an ILP sum-of-Linear-latency constraint. Fused QKV
and gate/up projections share decisions; prefill/decode choices are tied.
Main candidates are INT4/INT8/BF16; a separate FP4/INT8/BF16 control is included.
The 35 negative block-trace means out of 252 are retained, as are all samples.

DASO retains the original quality calibrator, Roofline latency model and
conservative dynamic program. Calibration uses 54 training and 18 held-out
policies per scenario, labeled on 100 WikiText **training** blocks. Local errors
cover all 36 layers in four depth buckets. Missing original Roofline JSON/NPZ
artifacts are rebuilt from the retained profiling CSVs in
`results/latency_models`; original source files are unchanged. Sparse FP4 decode
costs use M=32 because the runtime pads M=8 to that supported shape.

Test NLL uses 100 separate WikiText test blocks. All full likelihood tasks use
zero-shot evaluation and batch 4 (WikiText batch 1). Speed tests use B=8,
S=2048, O=1/64, two warmups and five measurements in one engine. Final speed and
allocator memory peaks use the same physical RTX 5090 (GPU 3); earlier parallel
measurements are retained under `speed_fixed_kv3gb`. These modes
share a fixed 3-GiB BF16 KV cache. Generation uses frozen original task samples
(CNN/DM 1000, DialogSum 1500, IWSLT 333), native Qwen non-thinking chat prompts,
left truncation at 3840 input tokens, greedy output up to 256 tokens, B=8,
and a fixed 6-GiB BF16 KV cache. ROUGE-L and SacreBLEU follow the original
evaluation definitions. Baselines and candidate policies use identical settings.

Full downstream candidates are selected from predicted budgets or ILP latency
sweeps **before** downstream scores are read. Every distinct sweep policy gets
real speed and independent test NLL; the predeclared representatives get all
full downstream tasks. Quality budgets and HAWQ latency limits have different
units; comparisons must use actual NLL and measured latency, not nominal budgets.

Measured quality constraints are imperfect: all 10 prefill search points exceed
their nominal budgets; seven of eight decode points meet their measured budgets.
The report includes every violation and a calibration-text re-evaluation of four
frozen representative policies. These diagnostics do not refit or replace policies.

All attempts, including failures and unfavorable measurements, remain here.
Large reproducible temporary tensors may live outside the run tree; their
locations and reconstruction commands are recorded in configuration files.
Do not delete code, logs, CSV files, documentation or result evidence.

Diagnostics include the unsuccessful W4A4 INT4 PTQ, FP16-cast Marlin INT4,
earlier batch-1 task attempts, initial high-variance HVPs and compiler errors.
They are kept and excluded from the final main-method table by the protocol.

Replaying uses the archived `configs/policies/*.json`, frozen input tensors and
JSONL prompts. Runtime Python is `/root/wja/miniconda3/envs/vllm/bin/python`;
source/model locations and versions are in `configs/environment.json`.
Set CUDA_VISIBLE_DEVICES to an available GPU and keep the recorded offline/cache
environment. `baseline_worker.py --policies ID --actions export,speed,nll,decode_nll,tasks,generation`
replays a uniform policy. `daso_pipeline.py` provides calibration/fitting and
`finish_queue.py` records the complete remaining evaluation stages. Command
manifests in `logs/` pin exact arguments, GPU and isolated code snapshots.
Temporary canonical states can be rebuilt by `prepare_sparse.py`; each released
candidate weight has a cleanup record and archived export metadata.

`results/export_metadata/*/tensor_footprint.json` describes serialized tensors;
it is separate from measured runtime peak GPU allocations. Original workflow
speed metadata can contain an outdated execution label; the invocation and
`last_memory_*.json` record actual engine reuse and the fixed cache override.

Reused implementation sources are archived in `reference/runtime_sources/`.
The exact cached likelihood evaluation splits (122 Arrow files, about 12 MB)
are frozen in `data/likelihood_datasets/`, with the recorded task configurations.
The original model file hashes and actual dependency versions are in `configs/`.

Plan: `dev/plans/141_qwen3_hawq_v3_rebuttal_plan.md`.
