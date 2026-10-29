#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
python "${ROOT}/workflows/paper/build_tables.py"
python "${ROOT}/workflows/paper/plots/tradeoff.py"
python "${ROOT}/workflows/paper/plots/calibration_sensitivity.py"
python "${ROOT}/workflows/paper/plots/quality_insight.py"
python "${ROOT}/workflows/paper/plots/pareto_audit.py" --scenario prefill_only
python "${ROOT}/workflows/paper/plots/pareto_audit.py" --scenario prefill_decode
mkdir -p "${ROOT}/results/paper/paper_source/Figures/draft2"
cp "${ROOT}/results/paper/figures/calibration_size_sensitivity.pdf" \
  "${ROOT}/results/paper/paper_source/Figures/draft2/calibration_size_sensitivity.pdf"
cp "${ROOT}/results/paper/figures/llama31_two_scenario_tradeoff_horizontal.pdf" \
  "${ROOT}/results/paper/paper_source/Figures/draft2/llama31_two_scenario_tradeoff_horizontal.pdf"
cp "${ROOT}/results/paper/figures/quality_modeling_insight.pdf" \
  "${ROOT}/results/paper/paper_source/Figures/draft2/quality_modeling_insight.pdf"
cp "${ROOT}/results/paper/appendix_pareto_audit/prefill_only/report/pareto_search_audit.pdf" \
  "${ROOT}/results/paper/paper_source/Figures/draft2/pareto_search_audit.pdf"
cp "${ROOT}/results/paper/appendix_pareto_audit/prefill_decode/report/pareto_search_audit_display.pdf" \
  "${ROOT}/results/paper/paper_source/Figures/draft2/pareto_search_audit_display.pdf"
python "${ROOT}/workflows/paper/build_result_manifest.py"
python "${ROOT}/workflows/paper/validate_archive.py"
