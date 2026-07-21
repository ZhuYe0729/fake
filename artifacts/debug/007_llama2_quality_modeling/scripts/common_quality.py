#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

DEBUG_ROOT = Path(__file__).resolve().parents[1]
FAKE_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE_ROOT = FAKE_ROOT.parent
SOURCE_003_ROOT = FAKE_ROOT / "artifacts/results/main/003_llama2_7b_arc_easy_accuracy"
SOURCE_003_SCRIPTS = SOURCE_003_ROOT / "scripts"

for path in (WORKSPACE_ROOT, FAKE_ROOT, SOURCE_003_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import (  # type: ignore  # noqa: E402
    CalibConfig,
    build_calib_loader,
    build_wikitext2_blocks,
    cleanup_cuda,
    compressible_modules,
    dtype_from_arg,
    load_model,
    model_spec,
    module_parent,
    utc_now,
)

MODEL_KEY = "llama2-7b"
CORE_METHODS = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4")
COMPRESSED_CORE_METHODS = ("dense_nvfp4", "sparse_bf16", "sparse_nvfp4")


@dataclass(frozen=True)
class QualityConfig:
    calib_samples: int = 32
    seq_len: int = 512
    seed: int = 0
    batch_size: int = 1
    cache_dir: str = "/home/agent/wja/.cache/huggingface"
    source_root: Path = SOURCE_003_ROOT
    output_root: Path = DEBUG_ROOT
    tokenizer_path: Path | None = None
    dataset_arrow_path: Path | None = None


def local_cuda_index(requested_gpu: int) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    count = torch.cuda.device_count()
    if count == 0:
        raise RuntimeError("CUDA is required")
    if requested_gpu < count:
        return requested_gpu
    if visible:
        return 0
    return requested_gpu


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_calibration_blocks(config: QualityConfig) -> tuple[torch.Tensor, dict[str, Any]]:
    calib = CalibConfig(
        samples=config.calib_samples,
        seq_len=config.seq_len,
        seed=config.seed,
        cache_dir=config.cache_dir,
    )
    blocks, metadata = build_wikitext2_blocks(
        calib,
        model_key=MODEL_KEY,
        tokenizer_path=config.tokenizer_path,
        dataset_arrow_path=config.dataset_arrow_path,
    )
    return blocks, metadata


def prepared_artifact(source_root: Path, method: str) -> Path:
    path = source_root / "prepared" / method / "model.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing prepared artifact for {method}: {path}. "
            "Run the 003 prepare script first or pass --source-root to an existing result root."
        )
    return path


def load_prepared_state(source_root: Path, method: str) -> dict[str, torch.Tensor]:
    payload = torch.load(prepared_artifact(source_root, method), map_location="cpu")
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"Prepared artifact for {method} has no state_dict")
    return state


def module_type(module_name: str) -> str:
    return module_name.rsplit(".", 1)[-1]


def layer_index(module_name: str) -> int:
    parts = module_name.split(".")
    if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
        return int(parts[2])
    return -1


def layer_bucket(index: int) -> str:
    if 0 <= index <= 7:
        return "layers_00_07"
    if 8 <= index <= 15:
        return "layers_08_15"
    if 16 <= index <= 23:
        return "layers_16_23"
    if 24 <= index <= 31:
        return "layers_24_31"
    return "other"


def module_family(module_name: str) -> str:
    typ = module_type(module_name)
    if typ in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return "attention"
    if typ in {"gate_proj", "up_proj", "down_proj"}:
        return "mlp"
    return "other"


def module_record(info: Any, index: int) -> dict[str, Any]:
    layer = layer_index(info.name)
    return {
        "module_index": index,
        "module_name": info.name,
        "layer": layer,
        "layer_bucket": layer_bucket(layer),
        "module_type": module_type(info.name),
        "module_family": module_family(info.name),
        "in_features": int(info.module.in_features),
        "out_features": int(info.module.out_features),
        "numel": int(info.module.weight.numel()),
    }


def tensor_stats(prefix: str, value: torch.Tensor) -> dict[str, Any]:
    x = value.detach().float()
    abs_x = x.abs()
    mean_abs = abs_x.mean().clamp(min=1e-12)
    return {
        f"{prefix}_mean": float(x.mean().item()),
        f"{prefix}_std": float(x.std(unbiased=False).item()),
        f"{prefix}_abs_mean": float(mean_abs.item()),
        f"{prefix}_abs_max": float(abs_x.max().item()),
        f"{prefix}_outlier_ratio_6x": float((abs_x > 6.0 * mean_abs).float().mean().item()),
    }


def weight_stats(module: nn.Linear) -> dict[str, Any]:
    w = module.weight.detach().float().cpu()
    rows = w.norm(dim=1)
    cols = w.norm(dim=0)
    out = tensor_stats("weight", w)
    out.update(
        {
            "weight_row_norm_mean": float(rows.mean().item()),
            "weight_row_norm_max": float(rows.max().item()),
            "weight_col_norm_mean": float(cols.mean().item()),
            "weight_col_norm_max": float(cols.max().item()),
        }
    )
    return out


def parse_policy(policy: str, module_names: Iterable[str]) -> set[str]:
    names = list(module_names)
    if policy == "none":
        return set()
    if policy == "all":
        return set(names)
    if policy.startswith("family:"):
        family = policy.split(":", 1)[1]
        return {name for name in names if module_family(name) == family}
    if policy.startswith("type:"):
        typ = policy.split(":", 1)[1]
        return {name for name in names if module_type(name) == typ}
    if policy.startswith("bucket:"):
        bucket = policy.split(":", 1)[1]
        return {name for name in names if layer_bucket(layer_index(name)) == bucket}
    if policy.startswith("layer:"):
        layer = int(policy.split(":", 1)[1])
        return {name for name in names if layer_index(name) == layer}
    if policy.startswith("layer_family:"):
        spec = policy.split(":", 1)[1]
        layer_text, family = spec.split(":", 1)
        layer = int(layer_text)
        return {name for name in names if layer_index(name) == layer and module_family(name) == family}
    if policy.startswith("module:"):
        module_name = policy.split(":", 1)[1]
        return {module_name}
    raise ValueError(f"Unsupported policy: {policy}")


def default_ablation_policies() -> list[str]:
    policies = [
        "all",
        "family:attention",
        "family:mlp",
        "type:q_proj",
        "type:k_proj",
        "type:v_proj",
        "type:o_proj",
        "type:gate_proj",
        "type:up_proj",
        "type:down_proj",
        "bucket:layers_00_07",
        "bucket:layers_08_15",
        "bucket:layers_16_23",
        "bucket:layers_24_31",
    ]
    for layer in (0, 4, 8, 12, 16, 20, 24, 28):
        policies.extend([f"layer:{layer}", f"layer_family:{layer}:attention", f"layer_family:{layer}:mlp"])
    return policies


def apply_compressed_weights(
    model: nn.Module,
    modules: list[Any],
    *,
    source_root: Path,
    method: str,
    selected_names: set[str],
) -> int:
    if method == "dense_bf16":
        return 0
    state = load_prepared_state(source_root, method)
    replaced = 0
    for info in modules:
        if info.name not in selected_names:
            continue
        key = f"{info.name}.weight"
        if key not in state:
            raise KeyError(f"{method} artifact missing {key}")
        parent, child_name = module_parent(model, info.name)
        module = getattr(parent, child_name)
        module.weight.data.copy_(state[key].to(device=module.weight.device, dtype=module.weight.dtype))
        bias_key = f"{info.name}.bias"
        if module.bias is not None and bias_key in state:
            module.bias.data.copy_(state[bias_key].to(device=module.bias.device, dtype=module.bias.dtype))
        replaced += 1
    return replaced


@torch.inference_mode()
def compute_nll(model: nn.Module, blocks: torch.Tensor, *, device: str, batch_size: int) -> dict[str, Any]:
    loader = build_calib_loader(blocks, batch_size=batch_size)
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        input_ids = batch.to(device=device, non_blocking=True)
        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits[:, :-1, :].float()
        labels = input_ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum")
        total_loss += float(loss.item())
        total_tokens += int(labels.numel())
    nll = total_loss / max(total_tokens, 1)
    return {
        "nll": nll,
        "ppl": float(math.exp(min(nll, 20.0))),
        "tokens": total_tokens,
        "loss_sum": total_loss,
    }


def load_llama_for_quality(*, device: str, dtype: torch.dtype) -> nn.Module:
    return load_model(MODEL_KEY, device=device, dtype=dtype)


def write_run_metadata(path: Path, payload: dict[str, Any]) -> None:
    spec = model_spec(MODEL_KEY)
    enriched = {
        "model_key": MODEL_KEY,
        "model_label": spec["label"],
        "model_path": spec["path"],
        "timestamp": utc_now(),
        **payload,
    }
    write_json(path, enriched)
