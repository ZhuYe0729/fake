#!/usr/bin/env python3
"""Resolve actual Llama2 prefill-only runtime checkpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEBUG = ROOT / "artifacts/debug/042_llama2_prefill_only_vllm_runtime_quality"
OUT = DEBUG / "manifest/policies.json"
MODEL = "/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf"
SUMMARY = ROOT / "artifacts/exports/vllm/ours/llama2-7b-chat/pareto_summary/summary.md"
UNIFORM_ROOT = ROOT / "artifacts/exports/vllm/baselines/llama2-7b-chat/checkpoints"
POLICY_ROOT = ROOT / "artifacts/debug/034_llama2_7b_chat_wikitext_pareto_solver/prefill_only/pareto/policies"
CHECKPOINTS = {
    4: ROOT / "artifacts/debug/034_llama2_7b_chat_wikitext_pareto_solver/validation/prefill_only/checkpoints/point_004",
    6: ROOT / "artifacts/debug/037_llama2_prefill_only_pareto/checkpoints/point_006",
    8: ROOT / "artifacts/debug/034_llama2_7b_chat_wikitext_pareto_solver/validation/prefill_only/checkpoints/point_008",
    9: ROOT / "artifacts/debug/037_llama2_prefill_only_pareto/checkpoints/point_009",
    11: ROOT / "artifacts/debug/037_llama2_prefill_only_pareto/checkpoints/point_011",
    12: ROOT / "artifacts/debug/034_llama2_7b_chat_wikitext_pareto_solver/validation/prefill_only/checkpoints/point_012",
    13: ROOT / "artifacts/debug/037_llama2_prefill_only_pareto/checkpoints/point_013",
    15: ROOT / "artifacts/debug/037_llama2_prefill_only_pareto/checkpoints/point_015",
    16: ROOT / "artifacts/debug/034_llama2_7b_chat_wikitext_pareto_solver/validation/prefill_only/checkpoints/point_016",
}
UNIFORM = {
    "dense_bf16": Path(MODEL),
    "dense_nvfp4": UNIFORM_ROOT / "uniform_dense_nvfp4",
    "marlin_nvfp4": UNIFORM_ROOT / "uniform_marlin_nvfp4",
    "sparse_bf16": UNIFORM_ROOT / "uniform_sparse_bf16",
    "sparse_nvfp4": UNIFORM_ROOT / "uniform_sparse_nvfp4",
}


def rows() -> list[dict[str, str]]:
    lines = SUMMARY.read_text().splitlines()
    head = next(i for i, line in enumerate(lines) if line.startswith("| scenario |"))
    keys = [x.strip() for x in lines[head].strip().strip("|").split("|")]
    out = []
    for line in lines[head + 2 :]:
        if not line.startswith("|"):
            break
        item = dict(zip(keys, [x.strip() for x in line.strip().strip("|").split("|")]))
        if item["scenario"].startswith("prefill-only"):
            out.append(item)
    return out


def normalized(path: Path) -> dict:
    item = json.loads(path.read_text())
    item.setdefault("default_prefill_method", item.pop("prefill_method", "dense_bf16"))
    item.setdefault("default_decode_method", item.pop("decode_method", "dense_bf16"))
    item.setdefault("modules_to_not_convert", ["lm_head"])
    item.setdefault("method_map", {})
    return item


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    entries = []
    for row in rows():
        label = row["policy"]
        entry = {"label": label, "family": row["family"], "recommended_use": row["recommended use"],
                 "e2e_ms": float(row["E2E ms"]), "speedup": float(row["speedup"]), "speed_source": row["speed source"]}
        if label in UNIFORM:
            checkpoint = UNIFORM[label]
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            entry.update(kind="uniform", checkpoint=rel(checkpoint))
        else:
            idx = int(label.rsplit("_", 1)[1])
            policy = POLICY_ROOT / f"point_{idx:03d}.json"
            checkpoint = CHECKPOINTS[idx]
            actual = checkpoint / "phase_hetero_policy.json"
            if normalized(policy) != normalized(actual):
                raise RuntimeError(f"policy mismatch: {policy} != {actual}")
            entry.update(kind="ours", checkpoint=rel(checkpoint), policy_json=rel(policy),
                         policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest())
        entries.append(entry)
    if len(entries) != 14:
        raise RuntimeError(f"expected 14 entries, got {len(entries)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"model_path": MODEL, "policies": entries}, indent=2) + "\n")
    print(f"wrote {OUT} ({len(entries)} policies)")


if __name__ == "__main__":
    main()
