#!/usr/bin/env python3
"""Collect phase-local errors with the canonical sparse runtime wrappers."""
from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).parent))
from scenario import EXP, MODEL, CANONICAL, CUTLASS, INPUT_TOKENS, OUTPUT_TOKENS
LINEARS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--method", choices=("dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "w4a16_ours"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--module-chunk-size", type=int, default=16)
    return parser.parse_args()


def bucket(name: str) -> int:
    return int(name.split(".")[2]) // 8


def fused_type(name: str) -> str:
    typ = name.rsplit(".", 1)[-1]
    if typ in {"q_proj", "k_proj", "v_proj"}:
        return "qkv_proj"
    if typ in {"gate_proj", "up_proj"}:
        return "gate_up_proj"
    return typ


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(CUTLASS))
    from cutlass_wrapper import (MarlinNVFP4Linear, NVFP4Linear, SparseBF16Linear, SparseNVFP4Linear,
                                 pack_marlin_nvfp4_weight, pack_sparse_bf16_weight,
                                 quantize_sparse_weight_bf16, quantize_weight_bf16)

    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"
    blocks = torch.load(EXP / "samples/wikitext_2048_64.pt", map_location="cpu", weights_only=True)[:args.blocks]
    state = (torch.load(CANONICAL / args.method / "model.pt", map_location="cpu", mmap=True, weights_only=True)["state_dict"]
             if args.method.startswith("sparse_") else None)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, local_files_only=True,
                                                  attn_implementation="eager").to(device).eval()
    modules = [(name, module) for name, module in model.named_modules()
               if name.rsplit(".", 1)[-1] in LINEARS and (state is None or f"{name}.weight" in state)]
    accum: dict[tuple[int, str], dict[str, float]] = {}

    for start in range(0, len(modules), args.module_chunk_size):
        chunk = modules[start:start + args.module_chunk_size]
        compressed = {}
        for name, module in chunk:
            weight = (state[f"{name}.weight"].to(device=device, dtype=torch.bfloat16)
                      if state is not None else module.weight.detach().to(dtype=torch.bfloat16).contiguous())
            if args.method == "sparse_bf16":
                compressed[name] = SparseBF16Linear(pack_sparse_bf16_weight(weight, prune=False)).eval()
            elif args.method == "sparse_nvfp4":
                compressed[name] = SparseNVFP4Linear(quantize_sparse_weight_bf16(weight, prune=False)).eval()
            elif args.method == "dense_nvfp4":
                compressed[name] = NVFP4Linear(quantize_weight_bf16(weight)).eval()
            else:
                compressed[name] = MarlinNVFP4Linear(pack_marlin_nvfp4_weight(weight)).eval()
            del weight

        def hook(name: str):
            def run(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
                x, y = inputs[0], output
                if args.phase == "decode":
                    x, y = x[:, -OUTPUT_TOKENS:], y[:, -OUTPUT_TOKENS:]
                if args.phase == "decode" and args.method == "sparse_nvfp4":
                    # The SM120 kernel requires a 32-aligned token dimension.
                    # Zero padding preserves the activation magnitude and the
                    # reported error is cropped back to the real decode tokens.
                    padding = (-x.shape[1]) % 32
                    padded = torch.cat((x, x.new_zeros((x.shape[0], padding, x.shape[2]))), dim=1)
                    estimate = compressed[name](padded)[:, :x.shape[1]]
                else:
                    estimate = compressed[name](x)
                error = estimate.float() - y.float()
                row = accum.setdefault((bucket(name), fused_type(name)), {"sse": 0.0, "ref": 0.0, "count": 0.0})
                row["sse"] += float(error.square().sum().item())
                row["ref"] += float(y.float().square().sum().item())
                row["count"] += float(y.numel())
            return run

        handles = [module.register_forward_hook(hook(name)) for name, module in chunk]
        try:
            with torch.inference_mode():
                for block in blocks:
                    ids = (block[:INPUT_TOKENS] if args.phase == "prefill" else block).unsqueeze(0).to(device)
                    model(input_ids=ids, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        del compressed
        gc.collect(); torch.cuda.empty_cache()

    rows = [{"phase": args.phase, "method": args.method, "layer_bucket": layer_bucket,
             "fused_type": typ, "blocks": args.blocks, "input_tokens": INPUT_TOKENS,
             "output_tokens": OUTPUT_TOKENS,
             "output_rel_mse": values["sse"] / max(values["ref"], 1e-12),
             "output_mse": values["sse"] / max(values["count"], 1.0),
             "output_count": int(values["count"]), "backend": "canonical_phase_wrapper"}
            for (layer_bucket, typ), values in sorted(accum.items())]
    output = EXP / "local_errors" / f"{args.phase}_{args.method}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()

