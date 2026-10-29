#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from daso.reporting.provenance import sha256


SOURCE_MAP = {
    "main_rtx5090/llama2_7b_chat": "060_two_model_two_scenario_result_consolidation",
    "main_rtx5090/llama31_8b_instruct/prefill_decode": "060_two_model_two_scenario_result_consolidation",
    "main_rtx5090/llama31_8b_instruct/prefill_only": "068_llama31_prefill_uniform_coverage_refinement",
    "cross_platform_pro6000": "068_two_model_two_scenario_pro6000_result_consolidation",
    "ablations": "067_llama31_ablation_consolidation + 070_quality_modeling_insight_figure",
    "appendix_pareto_audit": "069_llama31_pareto_search_audit",
    "modeling_calibration/llama2_7b_chat/prefill_only/latency": "regenerated with final external KernelLatencyPredictor",
    "modeling_calibration/llama2_7b_chat/prefill_only": "054_llama2_prefill_canonical_sparse_quality_recalibration",
    "modeling_calibration/llama2_7b_chat/prefill_decode": "056_llama2_prefill_decode_b8_o64_canonical_pareto",
    "modeling_calibration/llama31_8b_instruct/prefill_only/latency": "038_llama31_prefill_only + 061 warmed speed revalidation",
    "modeling_calibration/llama31_8b_instruct/prefill_only": "058_llama31_prefill_canonical_sparse_quality_recalibration",
    "modeling_calibration/llama31_8b_instruct/prefill_decode": "057_llama31_8b_instruct_b8_o64_canonical_pareto",
    "paper_source": "artifacts/paper",
}


def source_for(path: str) -> str:
    matches = [(prefix, source) for prefix, source in SOURCE_MAP.items() if path.startswith(prefix)]
    return max(matches, key=lambda item: len(item[0]))[1] if matches else "final paper assembly"


def provenance_level(path: Path) -> str:
    if path.suffix in {".png", ".pdf", ".tex", ".md"}:
        return "derived"
    if any(part in {"report", "summary", "tables", "figures"} for part in path.parts):
        return "aggregated"
    return "raw-structured"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] / "results/paper")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "manifest.json"
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        relative = path.relative_to(root)
        files.append({"path": str(relative), "bytes": path.stat().st_size,
                      "sha256": sha256(path), "provenance": provenance_level(relative),
                      "original_source": source_for(str(relative))})
    payload = {
        "schema_version": 1,
        "paper_definition": ["paper_source/wu.tex", "paper_source/appendix.tex"],
        "excluded": ["model weights", "canonical checkpoints", "datasets", "extension caches",
                     "temporary exports", "failed/intermediate runs"],
        "known_limitations": [
            "Llama2 generation BF16/dense-NVFP4/sparse-BF16 retain aggregate measurements but not all per-sample outputs.",
            "The manually authored motivation.pdf and overview3.pdf were not present in the source repository.",
        ],
        "files": files,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"files": len(files), "bytes": sum(item["bytes"] for item in files)}, indent=2))


if __name__ == "__main__":
    main()
