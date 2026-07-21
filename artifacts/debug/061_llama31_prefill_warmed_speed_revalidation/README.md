# Llama3.1 prefill warmed E2E speed revalidation

This experiment corrects only the E2E speed protocol used by 058.  It reuses
058's policies, canonical sparse assets, local latency predictions, and
quality model. It never writes into 058.

Before GPU execution, build the immutable 12-policy design:

```bash
python scripts/build_speed_design.py
```

When GPUs are idle, run only one policy per GPU:

```bash
python scripts/run_speed_anchors.py --gpus 1,2,3,4
python scripts/fit_speed_calibrator.py
python scripts/solve_pareto.py
```

Every policy uses B=8, input=2048, `max_num_seqs=8`,
`max_num_batched_tokens=16384`, an explicit prefill phase, one same-engine
warmup request, and five same-engine timed requests. The historical 058 speed
JSON files are intentionally not read as labels.

Checkpoint export is serialized to protect shared I/O. It loads a canonical
sparse state only when the policy actually uses that sparse method; uniform
dense policies do not pay the cost of loading unused SparseGPT states.
Dense-NVFP4 modules reuse the verified uniform dense-NVFP4 packed tensors,
instead of rerunning online packing during every mixed-policy export.

After the calibration is available, reuse the existing quality-constrained
solver logic with this new `calibration.csv`. Close any selected policy with:

```bash
python scripts/run_closure_speed_policy.py --policy point_000 --gpu 1
```

The wrapper imports the unchanged 058 solver formulation but resolves its
assets from 061, so it reads the warmed `calibration.csv` and writes new policy
JSONs under 061. `build_speed_design.py` copies only the already-fitted quality
model file needed by that solver; it does not refit quality.
