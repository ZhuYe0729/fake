#!/usr/bin/env python3
"""Measure actual wrapper output error for canonical sparse Llama3.1 modules."""
from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


from common import CUTLASS, MODEL, RUN

EXPERIMENT = RUN
LINEARS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("sparse_bf16", "sparse_nvfp4"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--module-chunk-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(CUTLASS))
    from cutlass_wrapper import (SparseBF16Linear, SparseNVFP4Linear,
                                 pack_sparse_bf16_weight,
                                 quantize_sparse_weight_bf16)
    state_path = EXPERIMENT / f"canonical/prepared/{args.method}/model.pt"
    state = torch.load(state_path, map_location="cpu", mmap=True, weights_only=True)["state_dict"]
    blocks = torch.load(EXPERIMENT / "samples/wikitext_2048_targets.pt", map_location="cpu")[:args.blocks]
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                  local_files_only=True,
                                                  attn_implementation="eager").to(device).eval()
    modules = [(name, module) for name, module in model.named_modules()
               if name.rsplit(".", 1)[-1] in LINEARS and f"{name}.weight" in state]
    rows = []
    for start in range(0, len(modules), args.module_chunk_size):
        chunk = modules[start:start + args.module_chunk_size]
        compressed: dict[str, torch.nn.Module] = {}
        totals = {name: {"sse": 0.0, "ref": 0.0} for name, _ in chunk}
        for name, _ in chunk:
            weight = state[f"{name}.weight"].to(device=device, dtype=torch.bfloat16)
            compressed[name] = (SparseBF16Linear(pack_sparse_bf16_weight(weight, prune=False)).eval()
                                if args.method == "sparse_bf16"
                                else SparseNVFP4Linear(quantize_sparse_weight_bf16(weight, prune=False)).eval())
            del weight

        def hook(name: str):
            def run(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
                estimate = compressed[name](inputs[0])
                totals[name]["sse"] += float((estimate.float() - output.float()).square().sum().item())
                totals[name]["ref"] += float(output.float().square().sum().item())
            return run

        handles = [module.register_forward_hook(hook(name)) for name, module in chunk]
        try:
            with torch.inference_mode():
                for block in blocks:
                    model(input_ids=block[:2048].unsqueeze(0).to(device), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        for name, _ in chunk:
            rows.append({"layer": int(name.split(".")[2]), "module_type": name.rsplit(".", 1)[-1],
                         "method": args.method,
                         "local_rel_mse": totals[name]["sse"] / max(totals[name]["ref"], 1e-12),
                         "blocks": args.blocks, "backend": "canonical_phase_wrapper"})
        del compressed, totals
        torch.cuda.empty_cache()
    output = EXPERIMENT / f"local_errors/{args.method}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(output)
    del model, state
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
