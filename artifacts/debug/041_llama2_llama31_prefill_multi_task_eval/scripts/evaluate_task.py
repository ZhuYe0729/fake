#!/usr/bin/env python3
"""Evaluate one manifest policy and one prefill-only lm-eval task."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[4]
DEBUG = ROOT / "artifacts/debug/041_llama2_llama31_prefill_multi_task_eval"
TASKS = ("wikitext", "winogrande", "arc_easy", "mmlu")
# RTX 5090 has sufficient memory for Llama2 rolling PPL at 2048 tokens, but
# Llama3.1-8B eager attention materialises a float32 score matrix that OOMs at
# that length.  Keep a fixed 1024-token context for every Llama3 policy so the
# PPL comparison remains internally identical and reproducible.
WIKITEXT_MAX_LENGTH = {"llama2-7b-chat": None, "llama3.1-8b-instruct": 1024}
STATE_ARTIFACT = {
    "dense_nvfp4": "dense_nvfp4",
    "sparse_bf16": "sparse_bf16",
    "sparse_nvfp4": "sparse_nvfp4",
    "w4a16_ours": "marlin_nvfp4",
    "marlin_nvfp4": "marlin_nvfp4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEBUG / "manifest/policies.json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", default="4")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0, help="logical CUDA index after CUDA_VISIBLE_DEVICES")
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def parent(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    obj = model
    for part in name.split(".")[:-1]:
        obj = getattr(obj, part)
    return obj, name.rsplit(".", 1)[-1]


def sources(fused: str) -> list[str]:
    base, kind = fused.rsplit(".", 1)
    if kind == "qkv_proj":
        return [base + suffix for suffix in (".q_proj", ".k_proj", ".v_proj")]
    if kind == "gate_up_proj":
        return [base + suffix for suffix in (".gate_proj", ".up_proj")]
    return [fused]


def method_map(entry: dict[str, Any]) -> dict[str, str]:
    if entry["kind"] == "ours":
        policy = json.loads((ROOT / entry["policy_json"]).read_text())
        return {name: item["prefill_method"] for name, item in policy["method_map"].items()}
    template = json.loads((ROOT / entry["policy_template"]).read_text())
    return {name: str(entry["uniform_method"]) for name in template["method_map"]}


def install_prefill_weights(model: nn.Module, methods: dict[str, str], prepared: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for method in sorted(set(methods.values())):
        names = [name for name, selected in methods.items() if selected == method]
        counts[method] = len(names)
        if method == "dense_bf16":
            continue
        artifact = STATE_ARTIFACT.get(method)
        if artifact is None:
            raise ValueError(f"unsupported prefill method: {method}")
        state_path = prepared / artifact / "model.pt"
        payload = torch.load(state_path, map_location="cpu")
        state = payload["state_dict"]
        for fused in names:
            for name in sources(fused):
                key = f"{name}.weight"
                if key not in state:
                    raise KeyError(f"{key} missing from {state_path}")
                obj, child = parent(model, name)
                layer = getattr(obj, child)
                if not isinstance(layer, nn.Linear):
                    raise TypeError(f"{name} is not nn.Linear: {type(layer)}")
                with torch.no_grad():
                    layer.weight.copy_(state[key].to(device=layer.weight.device, dtype=layer.weight.dtype))
        del state, payload
        gc.collect()
    return counts


def metric_values(payload: dict[str, Any], task: str) -> dict[str, Any]:
    selected = payload.get("results", {}).get(task) or payload.get("groups", {}).get(task) or {}
    prefixes = ("acc,", "acc_norm,", "word_perplexity,", "byte_perplexity,", "bits_per_byte,")
    return {key: value for key, value in selected.items() if key.startswith(prefixes)}


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    specs = manifest["models"]
    if args.model not in specs:
        raise KeyError(f"unknown model: {args.model}")
    entry = next((item for item in manifest["policies"] if item["model"] == args.model and item["label"] == args.policy), None)
    if entry is None:
        raise KeyError(f"unknown policy: {args.model}:{args.policy}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation")
    model_path = Path(specs[args.model]["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation="eager"
        ).to(device).eval()
        counts = install_prefill_weights(model, method_map(entry), ROOT / specs[args.model]["prepared"])
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.tasks import TaskManager

        max_length = WIKITEXT_MAX_LENGTH[args.model] if args.task == "wikitext" else None
        lm = HFLM(pretrained=model, tokenizer=str(model_path), backend="causal", dtype=torch.bfloat16,
                  device=device, batch_size=args.batch_size, max_length=max_length, trust_remote_code=False)
        result = lm_eval.simple_evaluate(
            model=lm, tasks=[args.task], num_fewshot=0, batch_size=args.batch_size,
            limit=args.limit, log_samples=False, random_seed=args.seed,
            numpy_random_seed=args.seed, torch_random_seed=args.seed,
            fewshot_random_seed=args.seed, task_manager=TaskManager(),
        )
        if result is None:
            raise RuntimeError("lm_eval.simple_evaluate returned None")
        elapsed = time.perf_counter() - started
        row = {
            "model": args.model, "policy": args.policy, "task": args.task,
            "backend": "lm_eval.HFLM/transformers", "dtype": "bfloat16", "num_fewshot": 0,
            "batch_size": args.batch_size, "max_length": max_length, "limit": args.limit, "seed": args.seed,
            "method_counts": counts, "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed, "elapsed_minutes": elapsed / 60,
            "metrics": metric_values(result, args.task),
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                            "transformers": package_version("transformers"), "lm_eval": package_version("lm-eval"),
                            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
            "raw_lm_eval": result,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(row, indent=2, sort_keys=True, default=str) + "\n")
        print(json.dumps({key: row[key] for key in ("model", "policy", "task", "elapsed_seconds", "metrics")}, default=str), flush=True)
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
