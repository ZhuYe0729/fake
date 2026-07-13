#!/usr/bin/env python3
"""Create deterministic PMPD samples and phase-heterogeneous calibration policies."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[8]
PMPD_ROOT = ROOT / "references/pmpd_eval_kit"
MODEL_PATH = Path("/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf")
DATA_ROOT = Path("/home/agent/wja/data/datasets/flaxquant")
# The former Vicuna filter tokenizer is not present in this workspace.  The
# target Llama tokenizer keeps the same "long-enough translation" selection
# rule while remaining completely local/offline.
IWSLT_TOKENIZER = str(MODEL_PATH)
SCENARIOS = ("prefill_only", "prefill_decode")
METHODS = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "w4a16_ours")
TYPES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--model-path", type=Path, default=MODEL_PATH)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--samples-per-dataset", type=int, default=100)
    p.add_argument("--policies", type=int, default=30)
    p.add_argument("--seed", type=int, default=85)
    return p.parse_args()


def modules() -> list[str]:
    return [f"model.layers.{layer}.{part}.{typ}" for layer in range(32)
            for part, typ in (("self_attn", "qkv_proj"), ("self_attn", "o_proj"), ("mlp", "gate_up_proj"), ("mlp", "down_proj"))]


def policy(policy_id: str, index: int, scenario: str, seed: int) -> dict:
    rng = random.Random(f"{seed}:{scenario}:{index}")
    names = modules()
    # Five anchors make the method scale observable; remaining policies mix layer/type/segment effects.
    if index < len(METHODS):
        prefill = decode = {name: METHODS[index] for name in names}
        if scenario == "prefill_decode" and METHODS[index] == "sparse_nvfp4":
            decode = {name: "dense_nvfp4" for name in names}
        kind = f"uniform_{METHODS[index]}"
    else:
        kind = "mixed"
        prefill, decode = {}, {}
        for name in names:
            layer = int(name.split(".")[2]); typ = name.rsplit(".", 1)[-1]
            bucket = layer // 8
            # Stable structure plus a small random component prevents only uniform-ratio samples.
            base = (index + layer * 3 + TYPES.index(typ) * 5 + bucket) % len(METHODS)
            prefill[name] = METHODS[(base + (rng.random() < 0.25)) % len(METHODS)]
            if scenario == "prefill_only":
                decode[name] = prefill[name]
            else:
                dbase = (index * 2 + layer + TYPES.index(typ) * 3) % len(METHODS)
                decode[name] = METHODS[(dbase + (rng.random() < 0.25)) % len(METHODS)]
                # Sparse NVFP4 has no M=16 decode kernel for these fused
                # Llama shapes, so it is not a legal decode action.
                if decode[name] == "sparse_nvfp4": decode[name] = "dense_nvfp4"
    return {
        "policy_id": policy_id, "policy_kind": kind, "scenario": scenario,
        "default_prefill_method": "dense_bf16", "default_decode_method": "dense_bf16",
        "modules_to_not_convert": ["lm_head"],
        "method_map": {name: {"prefill_method": prefill[name], "decode_method": decode[name]} for name in names},
    }


def build_samples(out: Path, data_root: Path, count: int) -> None:
    sys.path.insert(0, str(PMPD_ROOT)); import pmpd_eval  # type: ignore
    all_rows = []
    for dataset in ("cnn_dm_1000", "dsum", "IWSLT"):
        qargs = argparse.Namespace(dataset=dataset, split="test", data_root=data_root, question_begin=None, question_end=None, iwslt_filter_tokenizer=IWSLT_TOKENIZER)
        questions = pmpd_eval.build_questions(qargs)
        for row in questions[:count]:
            ref = row.get("reference", "")
            if isinstance(ref, list): ref = "\n".join(map(str, ref))
            all_rows.append({"dataset": dataset, "question_id": row["question_id"], "prompt": row["prompt"], "reference": str(ref)})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in all_rows) + "\n")


def main() -> None:
    a = args(); build_samples(a.output_root / "samples" / "pmpd_100x3.jsonl", a.data_root, a.samples_per_dataset)
    for scenario in SCENARIOS:
        target = a.output_root / "policies" / scenario; target.mkdir(parents=True, exist_ok=True)
        manifest = []
        for index in range(a.policies):
            pid = f"p{index:02d}"; item = policy(pid, index, scenario, a.seed)
            (target / f"{pid}.json").write_text(json.dumps(item, indent=2, sort_keys=True) + "\n")
            manifest.append({"policy_id": pid, "split": "train" if index < 21 else "holdout", "policy_kind": item["policy_kind"], "path": str(target / f"{pid}.json")})
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("created samples and 30 policies per scenario")


if __name__ == "__main__": main()
