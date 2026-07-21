#!/usr/bin/env python3
"""Build the policy inventory from the existing prefill-only paper tables."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
DEBUG = ROOT / "artifacts/debug/041_llama2_llama31_prefill_multi_task_eval"
OUT = DEBUG / "manifest/policies.json"

SPECS = {
    "llama2-7b-chat": {
        "model_path": "/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf",
        "summary": ROOT / "artifacts/exports/vllm/ours/llama2-7b-chat/pareto_summary/summary.md",
        "arc_dir": ROOT / "artifacts/debug/037_llama2_prefill_only_pareto/arc_challenge/full",
        "template": "artifacts/debug/034_llama2_7b_chat_wikitext_pareto_solver/prefill_only/pareto/policies/point_004.json",
        "prepared": "artifacts/exports/vllm/baselines/llama2-7b-chat/prepared",
        "expected": 14,
    },
    "llama3.1-8b-instruct": {
        "model_path": "/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct",
        "summary": ROOT / "artifacts/exports/vllm/ours/llama3.1-8b-instruct/pareto_summary/summary.md",
        "arc_dir": ROOT / "artifacts/debug/038_llama31_8b_instruct_prefill_only_pareto/arc_challenge/full",
        "template": "artifacts/debug/038_llama31_8b_instruct_prefill_only_pareto/pareto/policies/point_003.json",
        "prepared": "artifacts/exports/vllm/baselines/llama3.1-8b-instruct/prepared",
        "expected": 12,
    },
}
UNIFORM = {"dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "marlin_nvfp4"}


def prefill_rows(path: Path) -> list[dict[str, str]]:
    lines = path.read_text().splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("| scenario |"))
    rows: list[dict[str, str]] = []
    for line in lines[header + 2 :]:
        if not line.startswith("|"):
            break
        values = [part.strip() for part in line.strip().strip("|").split("|")]
        row = dict(zip([part.strip() for part in lines[header].strip().strip("|").split("|")], values))
        if row["scenario"].startswith("prefill-only"):
            rows.append(row)
    return rows


def main() -> None:
    policies: list[dict[str, object]] = []
    models: dict[str, dict[str, object]] = {}
    for model_key, spec in SPECS.items():
        models[model_key] = {key: value for key, value in spec.items() if key not in {"summary", "arc_dir", "expected"}}
        selected = prefill_rows(spec["summary"])
        if len(selected) != spec["expected"]:
            raise RuntimeError(f"{model_key}: expected {spec['expected']} prefill rows, found {len(selected)}")
        for row in selected:
            label = row["policy"]
            item: dict[str, object] = {
                "model": model_key,
                "label": label,
                "family": row["family"],
                "recommended_use": row["recommended use"],
                "e2e_ms": float(row["E2E ms"]),
                "speedup": float(row["speedup"]),
                "speed_source": row["speed source"],
            }
            if label in UNIFORM:
                item.update(kind="uniform", uniform_method=label, policy_template=spec["template"])
            else:
                arc = spec["arc_dir"] / f"{label}.json"
                if not arc.exists():
                    raise FileNotFoundError(arc)
                item.update(kind="ours", policy_json=json.loads(arc.read_text())["policy_json"])
            policies.append(item)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"models": models, "policies": policies}, indent=2) + "\n")
    print(f"wrote {OUT}: {len(policies)} policies")


if __name__ == "__main__":
    main()
