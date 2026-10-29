#!/usr/bin/env python3
"""Generate canonical SparseGPT BF16 states used by every runtime policy."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from daso.compression import prune_weight, select_compressible_modules
from daso.reporting.provenance import sha256


@torch.inference_mode()
def collect_hessians(model: torch.nn.Module, modules: list[object], blocks: torch.Tensor,
                     device: str) -> dict[str, torch.Tensor]:
    hessians = {item.name: torch.zeros((item.columns, item.columns), device=device,
                                       dtype=torch.float32) for item in modules}
    counts = {item.name: 0 for item in modules}
    handles = []
    for item in modules:
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor,
                 *, name: str = item.name) -> None:
            flat = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            batch = flat.shape[0]
            hessians[name] *= counts[name] / (counts[name] + batch)
            counts[name] += batch
            hessians[name] += ((2.0 / counts[name]) ** 0.5 * flat).T @ ((2.0 / counts[name]) ** 0.5 * flat)
        handles.append(item.module.register_forward_hook(hook))
    try:
        for block in blocks:
            model(input_ids=block.unsqueeze(0).to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if any(count == 0 for count in counts.values()):
        raise RuntimeError("calibration inputs did not reach every selected module")
    return hessians


def layer_groups(modules: list[object]) -> list[list[object]]:
    output: dict[int, list[object]] = {}
    for item in modules:
        parts = item.name.split(".")
        if len(parts) < 3 or not parts[2].isdigit():
            continue
        output.setdefault(int(parts[2]), []).append(item)
    return [output[index] for index in sorted(output)]


def prepare(model_path: Path, samples: torch.Tensor, output: Path, pattern: str,
            device: str, block_size: int, percdamp: float) -> None:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="eager").to(device).eval()
    model.requires_grad_(False)
    modules = [item for item in select_compressible_modules(model, "llama")
               if item.kind == "linear" and item.name.rsplit(".", 1)[-1] in
               {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}]
    records = []
    for group in layer_groups(modules):
        hessians = collect_hessians(model, group, samples, device)
        for item in group:
            weight, _mask, metadata = prune_weight(
                item.module.weight.data, hessians[item.name], pattern=pattern,
                block_size=block_size, percdamp=percdamp, module_name=item.name)
            item.module.weight.data.copy_(weight)
            records.append({"module": item.name, **metadata})
        del hessians
        gc.collect(); torch.cuda.empty_cache()
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    payload = {"state_dict": state, "metadata": {
        "algorithm": "sparsegpt", "pattern": pattern, "quantized": False,
        "model_path": str(model_path), "sample_count": len(samples),
        "block_size": block_size, "percdamp": percdamp, "modules": records}}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    del model, state, payload
    gc.collect(); torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=0.01)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = "cuda:0"
    samples = torch.load(args.samples, map_location="cpu", weights_only=True)[:args.blocks, :2048]
    states = {}
    for method, pattern in (("sparse_bf16", "dense_2_4"),
                            ("sparse_nvfp4", "nvfp4_pair_2_4_over_8")):
        output = args.output_root / method / "model.pt"
        prepare(args.model_path, samples, output, pattern, device, args.block_size, args.percdamp)
        states[method] = {"state": str(output.relative_to(args.output_root)),
                          "sha256": sha256(output), "pruning": "sparsegpt", "quantized": False}
    (args.output_root / "canonical_manifest.json").write_text(
        json.dumps({"states": states}, indent=2) + "\n")


if __name__ == "__main__":
    main()

