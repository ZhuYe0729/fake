# Llama2 canonical sparse prefill quality recalibration

This isolated bundle replaces the direct `--prune` sparse path used by 053.
`sparse_bf16` and `sparse_nvfp4` are first calibrated with the baseline
SparseGPT/Hessian procedure.  The NVFP4 canonical state deliberately remains
in sparse BF16, so phase export performs the only NVFP4 conversion.

Run order:

```bash
conda activate cospaq
python artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/scripts/bootstrap.py
python artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/scripts/prepare_canonical_sparse.py --gpu 1
python artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/scripts/verify_canonical_sparse.py
python artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/scripts/collect_canonical_sparse_local_errors.py --method sparse_bf16 --gpu 1
python artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/scripts/collect_canonical_sparse_local_errors.py --method sparse_nvfp4 --gpu 1
python artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/scripts/assemble_local_error_table.py

conda activate vllm
python artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/scripts/prewarm_phase_extensions.py
python artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/scripts/run_all.py --gpus 1,2,3,4

conda activate cospaq
python artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/scripts/merge_nll.py
python artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/scripts/fit_and_report.py
```

The fixed policy/sample design is copied from 053.  The first NLL run must not
start until canonical-state verification and local-feature assembly have passed.
