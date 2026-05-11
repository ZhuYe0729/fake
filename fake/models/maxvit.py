from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class MaxViTVariant:
    variant: str
    model_id: str
    model_path: Path
    result_key: str


MAXVIT_VARIANTS: dict[str, MaxViTVariant] = {
    "tiny": MaxViTVariant(
        variant="tiny",
        model_id="timm/maxvit_tiny_tf_224.in1k",
        model_path=Path("/data/home/scxj523/run/wja/data/models/timm/maxvit_tiny_tf_224.in1k"),
        result_key="maxvit_tiny",
    ),
    "small": MaxViTVariant(
        variant="small",
        model_id="timm/maxvit_small_tf_224.in1k",
        model_path=Path("/data/home/scxj523/run/wja/data/models/timm/maxvit_small_tf_224.in1k"),
        result_key="maxvit_small",
    ),
    "base": MaxViTVariant(
        variant="base",
        model_id="timm/maxvit_base_tf_224.in1k",
        model_path=Path("/data/home/scxj523/run/wja/data/models/timm/maxvit_base_tf_224.in1k"),
        result_key="maxvit_base",
    ),
    "large": MaxViTVariant(
        variant="large",
        model_id="timm/maxvit_large_tf_512.in21k_ft_in1k",
        model_path=Path("/data/home/scxj523/run/wja/data/models/timm/maxvit_large_tf_512.in21k_ft_in1k"),
        result_key="maxvit_large",
    ),
}
MAXVIT_VARIANT_CHOICES = tuple(MAXVIT_VARIANTS.keys())
DEFAULT_MAXVIT_VARIANT = "tiny"
DEFAULT_MAXVIT_MODEL_PATH = Path(
    "/data/home/scxj523/run/wja/data/models/timm/maxvit_tiny_tf_224.in1k"
)


def get_maxvit_variant(variant: str = DEFAULT_MAXVIT_VARIANT) -> MaxViTVariant:
    try:
        return MAXVIT_VARIANTS[variant]
    except KeyError as exc:
        choices = ", ".join(MAXVIT_VARIANT_CHOICES)
        raise ValueError(f"Unsupported MaxViT variant: {variant}. Choices: {choices}") from exc


def maxvit_input_size(config: dict[str, Any]) -> tuple[int, int, int]:
    pretrained_cfg = config.get("pretrained_cfg", {})
    input_size = pretrained_cfg.get("input_size", [3, 224, 224])
    if len(input_size) != 3:
        raise ValueError(f"Expected 3D MaxViT input_size, got: {input_size}")
    return tuple(int(value) for value in input_size)


def load_maxvit_dense(
    model_path: str | Path | None = None,
    dtype: str = "auto",
    device: str | torch.device = "cuda",
    variant: str = DEFAULT_MAXVIT_VARIANT,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    import timm

    variant_info = get_maxvit_variant(variant)
    model_dir = Path(model_path) if model_path is not None else variant_info.model_path
    config = _load_config(model_dir)
    arch = config.get("architecture", "maxvit_tiny_tf_224")
    num_classes = int(config.get("num_classes", 1000))
    config.setdefault("model_id", variant_info.model_id)
    config.setdefault("model_variant", variant_info.variant)
    config.setdefault("result_key", variant_info.result_key)

    model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    state_dict = _load_state_dict(model_dir, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Failed to load MaxViT weights cleanly: missing={missing}, unexpected={unexpected}"
        )

    torch_dtype = resolve_dtype(dtype, model)
    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)
    model = model.to(device)
    model.eval()
    return model, config


def resolve_dtype(dtype: str, model: torch.nn.Module) -> torch.dtype | None:
    if dtype == "auto":
        return None
    if dtype == "fp32":
        return torch.float32
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype}")


def model_input_dtype(model: torch.nn.Module) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def _load_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    with config_path.open("r") as f:
        return json.load(f)


def _load_state_dict(model_dir: Path, map_location: str) -> dict[str, torch.Tensor]:
    safetensors_path = model_dir / "model.safetensors"
    bin_path = model_dir / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path), device=map_location)
    if bin_path.exists():
        state = torch.load(bin_path, map_location=map_location)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        return state
    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {model_dir}")
