#!/usr/bin/env python3
"""Collect canonical phase-local output relative-MSE features."""
from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path

import torch


LINEARS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def bucket(name: str) -> int:
    return int(name.split(".")[2]) // 8


def fused_type(name: str) -> str:
    suffix = name.rsplit(".", 1)[-1]
    if suffix in {"q_proj", "k_proj", "v_proj"}:
        return "qkv_proj"
    if suffix in {"gate_proj", "up_proj"}:
        return "gate_up_proj"
    return suffix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--cutlass-wrapper-path", type=Path, required=True)
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--method", choices=("dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "w4a16_ours"), required=True)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--input-tokens", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--module-chunk-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    sys.path.insert(0, str(args.cutlass_wrapper_path.resolve()))
    from cutlass_wrapper import (MarlinNVFP4Linear, NVFP4Linear, SparseBF16Linear,
        SparseNVFP4Linear, pack_marlin_nvfp4_weight, pack_sparse_bf16_weight,
        quantize_sparse_weight_bf16, quantize_weight_bf16)
    from transformers import AutoModelForCausalLM
    device = "cuda:0"
    blocks = torch.load(args.samples, map_location="cpu", weights_only=True)[:args.blocks]
    state = None
    if args.method.startswith("sparse_"):
        payload = torch.load(args.canonical_root / args.method / "model.pt", map_location="cpu",
                             mmap=True, weights_only=True)
        state = payload["state_dict"]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="eager").to(device).eval()
    modules = [(name, module) for name, module in model.named_modules()
               if name.rsplit(".", 1)[-1] in LINEARS and
               (state is None or f"{name}.weight" in state)]
    accum: dict[tuple[int, str], dict[str, float]] = {}
    for start in range(0, len(modules), args.module_chunk_size):
        chunk = modules[start:start + args.module_chunk_size]
        compressed = {}
        for name, module in chunk:
            weight = (state[f"{name}.weight"].to(device=device, dtype=torch.bfloat16)
                      if state is not None else module.weight.detach().to(torch.bfloat16).contiguous())
            if args.method == "sparse_bf16":
                compressed[name] = SparseBF16Linear(pack_sparse_bf16_weight(weight, prune=False)).eval()
            elif args.method == "sparse_nvfp4":
                compressed[name] = SparseNVFP4Linear(quantize_sparse_weight_bf16(weight, prune=False)).eval()
            elif args.method == "dense_nvfp4":
                compressed[name] = NVFP4Linear(quantize_weight_bf16(weight)).eval()
            else:
                compressed[name] = MarlinNVFP4Linear(pack_marlin_nvfp4_weight(weight)).eval()
        def make_hook(name: str):
            def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
                x, reference = inputs[0], output
                if args.phase == "decode":
                    x, reference = x[:, -args.output_tokens:], reference[:, -args.output_tokens:]
                if args.phase == "decode" and args.method == "sparse_nvfp4":
                    padding = (-x.shape[1]) % 32
                    padded = torch.cat((x, x.new_zeros((x.shape[0], padding, x.shape[2]))), 1)
                    estimate = compressed[name](padded)[:, :x.shape[1]]
                else:
                    estimate = compressed[name](x)
                error = estimate.float() - reference.float()
                row = accum.setdefault((bucket(name), fused_type(name)), {"sse": 0.0, "ref": 0.0, "count": 0.0})
                row["sse"] += float(error.square().sum())
                row["ref"] += float(reference.float().square().sum())
                row["count"] += reference.numel()
            return hook
        handles = [module.register_forward_hook(make_hook(name)) for name, module in chunk]
        try:
            with torch.inference_mode():
                for block in blocks:
                    length = args.input_tokens if args.phase == "prefill" else args.input_tokens + args.output_tokens
                    model(input_ids=block[:length].unsqueeze(0).to(device), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        del compressed
        gc.collect(); torch.cuda.empty_cache()
    rows = [{"phase": args.phase, "method": args.method, "layer_bucket": key[0],
             "fused_type": key[1], "blocks": args.blocks,
             "output_rel_mse": value["sse"] / max(value["ref"], 1e-12),
             "output_mse": value["sse"] / max(value["count"], 1.0),
             "output_count": int(value["count"]), "backend": "canonical_phase_wrapper"}
            for key, value in sorted(accum.items())]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()

