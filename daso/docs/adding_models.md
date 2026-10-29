# Adding another model

1. Add a model TOML with layer count and dimensions.
2. Implement or select a module selector returning stable module names.
3. Validate legal kernels for every phase and shape; unsupported actions must
   be absent before optimization.
4. Generate canonical sparse states once and record hashes.
5. Freeze calibration policies and token samples, then collect local errors and
   real-runtime ΔNLL labels.
6. Fit the quality model, profile the target GPU, build the action table, and
   solve a budget sweep.
7. Export and measure representative closure points before running full tasks.
8. Record every configuration, backend revision, policy hash, and raw result in
   a new manifest. Never inherit absolute paths or calibrated latency from a
   different GPU.

The optional vision adapters under `extras/` are intentionally isolated until
their workload and quality interfaces are promoted to the main workflow.

