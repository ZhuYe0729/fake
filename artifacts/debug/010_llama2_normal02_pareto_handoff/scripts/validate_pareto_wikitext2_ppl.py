#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F

from common_pareto import DEBUG_ROOT, FAKE_ROOT, read_csv, read_json, write_csv, write_json

PREPARED_SOURCE_ROOT = FAKE_ROOT / "artifacts/results/main/003_llama2_7b_arc_easy_accuracy"

QUALITY_SCRIPTS = DEBUG_ROOT.parent / "007_llama2_quality_modeling" / "scripts"
if str(QUALITY_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(QUALITY_SCRIPTS))

from common_quality import (  # type: ignore  # noqa: E402
    cleanup_cuda,
    compressible_modules,
    dtype_from_arg,
    load_llama_for_quality,
    load_prepared_state,
    local_cuda_index,
    model_spec,
    module_parent,
    utc_now,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Pareto policies on WikiText-2 test perplexity.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--source-root", type=Path, default=PREPARED_SOURCE_ROOT)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=0, help="0 means use the full tokenized split.")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--cache-dir", default="/home/agent/wja/.cache/huggingface")
    parser.add_argument("--points", default="all", help="'all' or comma-separated point indices.")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    local_gpu = local_cuda_index(args.gpu)
    torch.cuda.set_device(local_gpu)
    device = f"cuda:{local_gpu}"
    dtype = dtype_from_arg(args.dtype)
    output_csv = args.output_csv or args.output_root / "validation" / "wikitext2_ppl" / "pareto_wikitext2_test_ppl.csv"
    metadata_path = output_csv.with_suffix(".metadata.json")

    point_indices = select_point_indices(args)
    existing_rows = read_existing_rows(output_csv) if args.skip_existing else []
    done = {int(row["point_index"]) for row in existing_rows if row.get("point_index", "") != ""}
    blocks, dataset_metadata = build_wikitext2_eval_blocks(
        split=args.dataset_split,
        seq_len=args.seq_len,
        max_tokens=args.max_tokens,
        cache_dir=args.cache_dir,
    )
    rows = list(existing_rows)

    print(
        f"WikiText-2 PPL blocks={tuple(blocks.shape)} eval_tokens={dataset_metadata['eval_tokens']} "
        f"points={point_indices} gpu={args.gpu}"
    )
    for point_index in point_indices:
        if point_index in done:
            print(f"skipping existing point={point_index}")
            continue
        policy_path = find_policy_json(args.output_root, point_index)
        print(f"validating point={point_index} policy={policy_path}")
        policy = read_json(policy_path)
        model = load_llama_for_quality(device=device, dtype=dtype)
        replaced = apply_policy_weights(model, policy, args.source_root)
        nll = compute_wikitext2_nll(model, blocks, device=device, batch_size=args.batch_size)
        row: dict[str, Any] = {
            "point_index": point_index,
            "policy_json": str(policy_path),
            "replaced_modules": replaced,
            "quality_cost": policy["summary"]["quality_cost"],
            "predicted_latency_ms": policy["summary"]["latency_ms"],
            "wikitext2_split": args.dataset_split,
            "seq_len": args.seq_len,
            "nll": nll["nll"],
            "ppl": nll["ppl"],
            "tokens": nll["tokens"],
            "blocks": int(blocks.shape[0]),
            "loss_sum": nll["loss_sum"],
        }
        rows.append(row)
        rows.sort(key=lambda item: int(item["point_index"]))
        write_csv(output_csv, rows)
        del model
        gc.collect()
        cleanup_cuda()

    write_json(
        metadata_path,
        {
            "timestamp": utc_now(),
            "source_root": str(args.source_root),
            "output_root": str(args.output_root),
            "output_csv": str(output_csv),
            "gpu": args.gpu,
            "dtype": args.dtype,
            "points": point_indices,
            "dataset": dataset_metadata,
            "metric": "teacher_forced_next_token_nll_ppl",
            "note": "Perplexity is computed on contiguous fixed-length WikiText-2 blocks. The first token of each block is context-only.",
        },
    )
    print(f"wrote {len(rows)} rows to {output_csv}")


def read_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return read_csv(path)


def select_point_indices(args: argparse.Namespace) -> list[int]:
    if args.points == "all":
        matches = sorted((args.output_root / "pareto" / "policies").glob("point_*.json"))
        if not matches:
            raise FileNotFoundError(f"no policy json files under {args.output_root / 'pareto' / 'policies'}")
        return [parse_point_index(path) for path in matches]
    return [int(item) for item in args.points.split(",") if item.strip()]


def parse_point_index(path: Path) -> int:
    parts = path.name.split("_")
    if len(parts) < 2:
        raise ValueError(f"cannot parse point index from {path}")
    return int(parts[1])


def find_policy_json(output_root: Path, point_index: int) -> Path:
    matches = sorted((output_root / "pareto" / "policies").glob(f"point_{point_index:03d}_*.json"))
    if not matches:
        raise FileNotFoundError(f"no policy json for point {point_index}")
    return matches[0]


def build_wikitext2_eval_blocks(*, split: str, seq_len: int, max_tokens: int, cache_dir: str) -> tuple[torch.Tensor, dict[str, Any]]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    spec = model_spec("llama2-7b")
    tokenizer = AutoTokenizer.from_pretrained(
        spec["path"],
        local_files_only=True,
        use_fast=True,
        trust_remote_code=bool(spec["trust_remote_code"]),
    )
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split=split,
        cache_dir=cache_dir,
    )
    texts = [row["text"] for row in dataset if row.get("text") and row["text"].strip()]
    if not texts:
        raise RuntimeError(f"WikiText-2 split={split} returned no non-empty text")
    joined = "\n\n".join(texts)
    tokenized = tokenizer(joined, return_tensors="pt", add_special_tokens=False).input_ids[0].long()
    raw_tokens = int(tokenized.numel())
    if max_tokens > 0:
        tokenized = tokenized[:max_tokens]
    usable = int(tokenized.numel()) // seq_len * seq_len
    if usable < seq_len:
        raise RuntimeError(f"not enough tokens for one block: tokens={tokenized.numel()} seq_len={seq_len}")
    blocks = tokenized[:usable].reshape(-1, seq_len).contiguous()
    metadata = {
        "dataset_name": "Salesforce/wikitext",
        "dataset_config": "wikitext-2-raw-v1",
        "dataset_split": split,
        "cache_dir": cache_dir,
        "raw_tokens": raw_tokens,
        "max_tokens": max_tokens,
        "eval_tokens": usable,
        "seq_len": seq_len,
        "blocks": int(blocks.shape[0]),
    }
    return blocks, metadata


def apply_policy_weights(model, policy: dict[str, Any], source_root: Path) -> int:
    modules = {info.name: info for info in compressible_modules(model, "llama2-7b")}
    states: dict[str, dict[str, torch.Tensor]] = {}
    replaced = 0
    for item in policy["modules"]:
        module_name = item["module_name"]
        method = item["selected_prefill_backend"]
        if method == "dense_bf16":
            continue
        if method not in states:
            states[method] = load_prepared_state(source_root, method)
        state = states[method]
        info = modules[module_name]
        key = f"{module_name}.weight"
        if key not in state:
            raise KeyError(f"{method} artifact missing {key}")
        parent, child_name = module_parent(model, info.name)
        module = getattr(parent, child_name)
        module.weight.data.copy_(state[key].to(device=module.weight.device, dtype=module.weight.dtype))
        bias_key = f"{module_name}.bias"
        if module.bias is not None and bias_key in state:
            module.bias.data.copy_(state[bias_key].to(device=module.bias.device, dtype=module.bias.dtype))
        replaced += 1
    return replaced


@torch.inference_mode()
def compute_wikitext2_nll(model, blocks: torch.Tensor, *, device: str, batch_size: int) -> dict[str, Any]:
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, int(blocks.shape[0]), batch_size):
        batch = blocks[start : start + batch_size].to(device=device, non_blocking=True)
        outputs = model(input_ids=batch, use_cache=False)
        logits = outputs.logits[:, :-1, :].float()
        labels = batch[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum")
        total_loss += float(loss.item())
        total_tokens += int(labels.numel())
        del outputs, logits, labels, batch, loss
    nll = total_loss / max(total_tokens, 1)
    return {
        "nll": nll,
        "ppl": float(math.exp(min(nll, 20.0))),
        "tokens": total_tokens,
        "loss_sum": total_loss,
    }


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_cuda()
