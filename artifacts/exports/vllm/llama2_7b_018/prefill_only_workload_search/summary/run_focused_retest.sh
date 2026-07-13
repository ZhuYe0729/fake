#!/usr/bin/env bash
set -euo pipefail

source /home/agent/wja/miniconda3/etc/profile.d/conda.sh
conda activate cospaq

cd /root/wja/project/my/cospaq/fake

python artifacts/exports/vllm/llama2_7b_018/broad_grid_vllm/scripts/benchmark_broad_grid_vllm_parallel.py \
  --output-dir artifacts/exports/vllm/llama2_7b_018/prefill_only_workload_search/focused_uniform_vllm \
  --methods dense_bf16,dense_nvfp4,sparse_bf16,sparse_nvfp4,marlin_nvfp4 \
  --gpus "${GPUS:-0,1,2,3,4}" \
  --batches 1,2 \
  --input-seqs 128,256,512 \
  --output-seqs 1 \
  --warmup-iters 1 \
  --iters 5 \
  --continue-on-error

python artifacts/exports/vllm/llama2_7b_018/promising_policy_retest/scripts/solve_promising_policies.py \
  --scenarios artifacts/exports/vllm/llama2_7b_018/prefill_only_workload_search/summary/focused_retest_scenarios.csv \
  --output-dir artifacts/exports/vllm/llama2_7b_018/prefill_only_workload_search/policies

python artifacts/exports/vllm/llama2_7b_018/promising_policy_retest/scripts/export_promising_policy_checkpoints.py \
  --policy-dir artifacts/exports/vllm/llama2_7b_018/prefill_only_workload_search/policies/unique_policies \
  --output-dir artifacts/exports/vllm/llama2_7b_018/prefill_only_workload_search/checkpoints \
  --force

python artifacts/exports/vllm/llama2_7b_018/promising_policy_retest/scripts/benchmark_promising_policy_vllm_parallel.py \
  --policy-csv artifacts/exports/vllm/llama2_7b_018/prefill_only_workload_search/policies/scenario_policies.csv \
  --checkpoint-root artifacts/exports/vllm/llama2_7b_018/prefill_only_workload_search/checkpoints \
  --output-dir artifacts/exports/vllm/llama2_7b_018/prefill_only_workload_search/focused_hetero_vllm \
  --method-name focused_optimized_hetero \
  --output-prefix focused_optimized_hetero \
  --gpus "${GPUS:-0}" \
  --scenarios b1_in128_out1,b1_in256_out1,b2_in128_out1,b1_in512_out1,b2_in256_out1 \
  --warmup-iters 1 \
  --iters 5
