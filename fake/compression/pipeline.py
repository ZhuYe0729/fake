from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fake.compression.hessian import collect_hessian_diag
from fake.compression.int4 import INT4Config
from fake.compression.modules import flatten_weight, restore_weight_shape, select_compressible_modules
from fake.compression.nvfp4 import NVFP4Config, fake_quantize_nvfp4_weight
from fake.compression.pruning import (
    PruneResult,
    prune_dense_2_4,
    prune_nvfp4_pair_2_4,
    prune_unstructured,
)
from fake.compression.sparsegpt import sparsegpt_int4_compress_module


PRUNE_METHODS = {
    "unstructured_sparse",
    "semi_structured_sparse",
    "nvfp4_unstructured_sparse",
    "nvfp4_semi_structured_sparse",
    "nvfp4_4over6_unstructured_sparse",
    "nvfp4_4over6_semi_structured_sparse",
}
QUANT_METHODS = {
    "nvfp4",
    "int4",
    "nvfp4_unstructured_sparse",
    "nvfp4_semi_structured_sparse",
    "nvfp4_4over6_unstructured_sparse",
    "nvfp4_4over6_semi_structured_sparse",
}
INT4_SPARSEGPT_METHODS = {
    "int4_unstructured_sparse",
    "int4_semi_structured_sparse",
}
SUPPORTED_METHODS = sorted(PRUNE_METHODS | QUANT_METHODS | INT4_SPARSEGPT_METHODS)


@dataclass(frozen=True)
class CompressionConfig:
    model_name: str
    method: str
    calib_samples: int
    sparsity: float = 0.5
    nvfp4_group_size: int = 16
    nvfp4_scale_precision: str = "fp16"
    nvfp4_scale_rule: str = "static_6"
    nvfp4_scale_remap: str = "none"
    int4_group_size: int = 32
    int4_scale_precision: str = "fp16"
    sparsegpt_block_size: int = 128
    sparsegpt_percdamp: float = 0.01
    save_full_masks: bool = False
    save_full_scales: bool = False


def compress_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str | torch.device,
    input_dtype: torch.dtype,
    config: CompressionConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if config.method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported compression method: {config.method}")

    modules = select_compressible_modules(model, config.model_name)
    hessian_algo = _hessian_algo(config.model_name, config.method)
    print(f"[compression] selected_modules={len(modules)} hessian_algo={hessian_algo}")
    hessian_diag = None
    if config.method not in INT4_SPARSEGPT_METHODS:
        hessian_diag = collect_hessian_diag(
            model=model,
            modules=modules,
            dataloader=dataloader,
            device=device,
            input_dtype=input_dtype,
            max_samples=config.calib_samples,
        )

    masks: dict[str, Any] = {"format": "metadata_only", "modules": {}}
    scales: dict[str, Any] = {"format": "metadata_only", "modules": {}}
    module_records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for idx, info in enumerate(modules, start=1):
        print(f"[compression] {idx}/{len(modules)} {info.name}")
        matrix = flatten_weight(info.module)
        hdiag = hessian_diag.get(info.name) if hessian_diag is not None else None
        record: dict[str, Any] = {
            "name": info.name,
            "kind": info.kind,
            "columns": info.columns,
            "reason": info.reason,
            "prune": None,
            "quant": None,
        }

        if config.method in INT4_SPARSEGPT_METHODS:
            int4_config = INT4Config(
                group_size=config.int4_group_size,
                scale_precision=config.int4_scale_precision,
            )
            sparsegpt_result = sparsegpt_int4_compress_module(
                model=model,
                info=info,
                dataloader=dataloader,
                device=device,
                input_dtype=input_dtype,
                max_samples=config.calib_samples,
                method=config.method,
                sparsity=config.sparsity,
                int4_config=int4_config,
                block_size=config.sparsegpt_block_size,
                percdamp=config.sparsegpt_percdamp,
            )
            record["prune"] = sparsegpt_result.prune_stats
            record["quant"] = sparsegpt_result.quant_stats
            if sparsegpt_result.mask is None or sparsegpt_result.scales is None:
                skipped.append({"name": info.name, **sparsegpt_result.prune_stats})
            else:
                masks["modules"][info.name] = _mask_payload(
                    PruneResult(sparsegpt_result.weight, sparsegpt_result.mask, sparsegpt_result.prune_stats),
                    config.save_full_masks,
                )
                scales["modules"][info.name] = _scale_payload(
                    sparsegpt_result.scales,
                    sparsegpt_result.quant_stats,
                    config.save_full_scales,
                )
                matrix = sparsegpt_result.weight

        elif config.method in PRUNE_METHODS:
            assert hessian_diag is not None
            prune_result = _prune_matrix(matrix, hdiag, config)
            record["prune"] = prune_result.stats
            if prune_result.mask is None:
                skipped.append({"name": info.name, **prune_result.stats})
            else:
                masks["modules"][info.name] = _mask_payload(prune_result, config.save_full_masks)
                matrix = prune_result.weight

        if config.method in QUANT_METHODS:
            qconfig = NVFP4Config(
                group_size=config.nvfp4_group_size,
                scale_precision=config.nvfp4_scale_precision,
                scale_rule=config.nvfp4_scale_rule,
                scale_remap=config.nvfp4_scale_remap,
            )
            qresult = fake_quantize_nvfp4_weight(matrix, qconfig)
            record["quant"] = qresult.stats
            if qresult.scales is None:
                skipped.append({"name": info.name, **qresult.stats})
            else:
                scales["modules"][info.name] = _scale_payload(qresult.scales, qresult.stats, config.save_full_scales)
                matrix = qresult.weight

        info.module.weight.data.copy_(restore_weight_shape(info.module, matrix))
        module_records.append(record)

    metadata = {
        **asdict(config),
        "hessian_algo": hessian_algo,
        "selected_modules": len(modules),
        "compressed_modules": len(module_records),
        "skipped": skipped,
        "modules": module_records,
    }
    return metadata, masks, scales


def default_calib_samples(model_name: str) -> int:
    if model_name == "maxvit":
        return 128
    if model_name == "dinov3_vit7b16":
        return 16
    raise ValueError(f"Unsupported model: {model_name}")


def default_calib_batch_size(model_name: str) -> int:
    if model_name == "maxvit":
        return 16
    if model_name == "dinov3_vit7b16":
        return 1
    raise ValueError(f"Unsupported model: {model_name}")


def default_nvfp4_group_size(method: str) -> int:
    if method in ("nvfp4_semi_structured_sparse", "nvfp4_4over6_semi_structured_sparse"):
        return 32
    return 16


def default_int4_group_size(method: str) -> int:
    if method == "int4":
        return 32
    if method == "int4_unstructured_sparse":
        return 32
    if method == "int4_semi_structured_sparse":
        return 64
    return 32


def _hessian_algo(model_name: str, method: str) -> str:
    if method in INT4_SPARSEGPT_METHODS:
        return "sparsegpt_full_hessian"
    return "obc_diag" if model_name == "maxvit" else "sparsegpt_diag"


def _prune_matrix(matrix: torch.Tensor, hdiag: torch.Tensor | None, config: CompressionConfig) -> PruneResult:
    if config.method == "int4":
        raise ValueError("Method does not require pruning: int4")
    if config.method in ("unstructured_sparse", "nvfp4_unstructured_sparse", "nvfp4_4over6_unstructured_sparse"):
        return prune_unstructured(matrix, config.sparsity, hdiag)
    if config.method == "semi_structured_sparse":
        return prune_dense_2_4(matrix, hdiag)
    if config.method in ("nvfp4_semi_structured_sparse", "nvfp4_4over6_semi_structured_sparse"):
        return prune_nvfp4_pair_2_4(matrix, hdiag)
    raise ValueError(f"Method does not require pruning: {config.method}")


def _mask_payload(result: PruneResult, save_full: bool) -> dict[str, Any]:
    payload = dict(result.stats)
    if save_full and result.mask is not None:
        payload["mask"] = result.mask.cpu()
        payload["format"] = "full_bool"
    else:
        payload["format"] = "metadata_only"
    return payload


def _scale_payload(scales: torch.Tensor, stats: dict[str, Any], save_full: bool) -> dict[str, Any]:
    payload = dict(stats)
    if save_full:
        payload["scales"] = scales.cpu()
        payload["format"] = "full_tensor"
    else:
        payload["format"] = "metadata_only"
    return payload
