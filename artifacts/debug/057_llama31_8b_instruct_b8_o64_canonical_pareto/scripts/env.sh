#!/usr/bin/env bash
# Shared immutable protocol and paths for the Llama-3.1 B=8/O=64 experiment.
export COSPAQ_EXPERIMENT_ROOT="/root/wja/project/my/cospaq/fake/artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto"
export COSPAQ_EXPERIMENT_DIR="$COSPAQ_EXPERIMENT_ROOT/llama31_8b_instruct"
export COSPAQ_MODEL_PATH="/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
export COSPAQ_CANONICAL_DIR="$COSPAQ_EXPERIMENT_DIR/canonical/prepared"
export COSPAQ_BASELINE_DIR="/root/wja/project/my/cospaq/fake/artifacts/exports/vllm/baselines/llama3.1-8b-instruct"
export COSPAQ_MODEL_LABEL="Llama-3.1-8B-Instruct"
export COSPAQ_EXPORTER="$COSPAQ_EXPERIMENT_ROOT/scripts/export_phase_checkpoint.py"
export COSPAQ_SPEED_RUNNER="/root/wja/project/my/cospaq/fake/artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/scripts/run_speed_policy.sh"
export COSPAQ_VERIFY_CHECKPOINT="/root/wja/project/my/cospaq/fake/artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/scripts/verify_canonical_checkpoint.py"
