# DINOv3 Seeded No-Checkpoint Eval Plan

## Goal

Add a DINOv3 ViT-7B/16 multi-seed accuracy path that evaluates compressed candidates without saving full 26GB fake-quant checkpoints for every seed.

## Plan

1. Add a Python script that loads dense DINOv3, builds a seeded/shuffled calibration loader, runs `compress_model` in memory, evaluates ImageNet accuracy, and appends results to CSV.
2. For 4/6 methods, support evaluating activation quant off and/or on after one in-memory compression pass.
3. Add a Slurm wrapper that can sweep `METHODS` and `CALIB_SEEDS`.
4. Record enough CSV metadata to identify method, seed, calibration settings, scale rule, group size, and activation quant state.
5. Run syntax checks.

## Notes

- No `model.pt`, `masks.pt`, or `scales.pt` files are written by this path.
- After picking a best seed, use the existing checkpoint prepare path only for the final selected candidate if a persistent checkpoint is needed.
