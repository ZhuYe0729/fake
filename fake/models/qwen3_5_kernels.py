from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from fake.kernels.cutlass_nvfp4 import (
    CutlassNVFP4Config,
    ReplacementReport,
    _load_cutlass_nvfp4_symbols,
    replace_linear_with_cutlass_nvfp4,
)
from fake.kernels.cutlass_sparse_bf16 import (
    CutlassSparseBF16Config,
    PaddedSparseBF16Linear,
    SparseBF16ReplacementReport,
    SPARSE_BF16_BLOCKED_SHAPES,
    _load_cutlass_sparse_bf16_symbols,
    replace_linear_with_cutlass_sparse_bf16,
)
from fake.kernels.cutlass_sparse_nvfp4 import (
    CutlassSparseNVFP4Config,
    PaddedSparseNVFP4Linear,
    SparseReplacementReport,
    _load_cutlass_sparse_nvfp4_symbols,
    replace_linear_with_cutlass_sparse_nvfp4,
)
from fake.kernels.marlin_nvfp4 import (
    MARLIN_CHECKPOINT_FORMAT,
    MarlinNVFP4Config,
    MarlinReplacementReport,
    _make_marlin_workspace,
    install_marlin_nvfp4_modules_from_state_dict,
    prepare_marlin_nvfp4_packed_model,
)
from fake.kernels.offline_hybrid_policy import load_policy_json


QWEN3_5_KERNEL_CHECKPOINT_FORMAT = "qwen3_5_kernel_packed_v1"
QWEN3_5_HYBRID_NVFP4_CHECKPOINT_FORMAT = "qwen3_5_hybrid_nvfp4_packed_v1"
QWEN3_5_REAL_KERNEL_METHODS = (
    "dense",
    "dense_nvfp4",
    "sparse_bf16",
    "sparse_nvfp4",
    "marlin_nvfp4",
    "hybrid_nvfp4",
    "hybrid_nvfp4_major",
    "manual_hybrid_m1",
    "manual_hybrid_m4",
    "manual_hybrid_m8",
    "manual_hybrid_m16",
    "shape_workload_hybrid",
    "predictor_hybrid",
)
QWEN3_5_HYBRID_NVFP4_METHODS = (
    "hybrid_nvfp4",
    "hybrid_nvfp4_major",
)
QWEN3_5_MANUAL_HYBRID_METHODS = (
    "manual_hybrid_m1",
    "manual_hybrid_m4",
    "manual_hybrid_m8",
    "manual_hybrid_m16",
)
QWEN3_5_MANUAL_HYBRID_CHECKPOINT_FORMAT = "qwen3_5_manual_hybrid_packed_v1"
QWEN3_5_SWH_CHECKPOINT_FORMAT = "qwen3_5_shape_workload_hybrid_packed_v1"
QWEN3_5_SWH_METHODS = ("shape_workload_hybrid",)
QWEN3_5_PREDICTOR_HYBRID_CHECKPOINT_FORMAT = "qwen3_5_predictor_hybrid_packed_v1"
QWEN3_5_PREDICTOR_HYBRID_METHODS = ("predictor_hybrid",)


@dataclass(frozen=True)
class QwenManualHybridReplacementReport:
    hybrid_scheme: str
    config: dict[str, Any]
    replaced_linear_count: int
    skipped_linear_count: int
    skipped: list[dict[str, str]]
    backend_counts: dict[str, int]

    @property
    def backend(self) -> str:
        return f"qwen_manual_{self.hybrid_scheme}"

    def csv_fields(self) -> dict[str, object]:
        return {
            "kernel_backend": self.backend,
            "nvfp4_block_size": "manual",
            "nvfp4_backend": "manual_per_linear",
            "nvfp4_quant_backend": "manual_per_linear",
            "nvfp4_sf_layout": "manual_per_linear",
            "marlin_activation_dtype": str(self.config.get("decode_activation_dtype", "torch.bfloat16")).replace("torch.", ""),
            "replaced_linear_count": self.replaced_linear_count,
            "skipped_linear_count": self.skipped_linear_count,
        }

@dataclass(frozen=True)
class QwenHybridNVFP4ReplacementReport:
    backend: str
    config: dict[str, Any]
    replaced_linear_count: int
    skipped_linear_count: int
    skipped: list[dict[str, str]]

    def csv_fields(self) -> dict[str, object]:
        decode_activation_dtype = self.config.get("decode_activation_dtype", "torch.bfloat16")
        return {
            "kernel_backend": self.backend,
            "nvfp4_block_size": "16/32",
            "nvfp4_backend": "hybrid_cutlass_dense_w4a4_marlin_w4a16",
            "nvfp4_quant_backend": "shared_canonical_nvfp4",
            "nvfp4_sf_layout": "canonical_lazy_cutlass_marlin",
            "marlin_activation_dtype": str(decode_activation_dtype).replace("torch.", ""),
            "replaced_linear_count": self.replaced_linear_count,
            "skipped_linear_count": self.skipped_linear_count,
        }


Report = (
    ReplacementReport
    | SparseBF16ReplacementReport
    | SparseReplacementReport
    | MarlinReplacementReport
    | QwenHybridNVFP4ReplacementReport
    | QwenManualHybridReplacementReport
)


class QwenHybridDenseNVFP4Linear(nn.Module):
    """Dense NVFP4 Linear sharing one canonical weight across W4A4 and W4A16."""

    in_features: int
    out_features: int

    def __init__(
        self,
        canonical: Any,
        *,
        decode_activation_dtype: torch.dtype = torch.bfloat16,
        marlin_m_threshold: int = 16,
        prefill_backend: str = "dense_nvfp4",
        decode_backend: str = "marlin_nvfp4",
    ) -> None:
        super().__init__()
        self.in_features = int(canonical.in_features)
        self.out_features = int(canonical.out_features)
        self.original_dtype = canonical.original_dtype
        self.decode_activation_dtype = decode_activation_dtype
        self.marlin_m_threshold = int(marlin_m_threshold)
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend
        self.register_buffer("canonical_packed_weight", canonical.packed_weight.contiguous(), persistent=True)
        self.register_buffer("canonical_logical_scale", canonical.logical_scale.contiguous(), persistent=True)
        self.register_buffer("canonical_global_scale", canonical.global_scale.contiguous(), persistent=True)
        if canonical.bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", canonical.bias.contiguous(), persistent=True)
        object.__setattr__(self, "_cutlass_linear", None)
        object.__setattr__(self, "_marlin_linear", None)
        self.eval()
        self.requires_grad_(False)

    def _canonical(self) -> Any:
        wrapper = _load_wrapper()
        return wrapper.NVFP4CanonicalWeight(
            packed_weight=self.canonical_packed_weight,
            logical_scale=self.canonical_logical_scale,
            global_scale=self.canonical_global_scale,
            in_features=self.in_features,
            out_features=self.out_features,
            original_dtype=self.original_dtype,
            bias=self.bias,
        )

    def _get_cutlass_linear(self) -> nn.Module:
        if self._cutlass_linear is None:
            wrapper = _load_wrapper()
            object.__setattr__(
                self,
                "_cutlass_linear",
                wrapper.NVFP4Linear(wrapper.canonical_to_cutlass_nvfp4_weight(self._canonical())).eval(),
            )
        return self._cutlass_linear

    def _get_marlin_linear(self) -> nn.Module:
        if self._marlin_linear is None:
            wrapper = _load_wrapper()
            object.__setattr__(
                self,
                "_marlin_linear",
                wrapper.MarlinNVFP4Linear(
                    wrapper.canonical_to_marlin_nvfp4_weight(
                        self._canonical(),
                        activation_dtype=self.decode_activation_dtype,
                    )
                ).eval(),
            )
        return self._marlin_linear

    def forward_w4a4(self, x: torch.Tensor) -> torch.Tensor:
        return self._get_cutlass_linear()(x)

    def forward_w4a16(self, x: torch.Tensor) -> torch.Tensor:
        return self._get_marlin_linear()(x)

    def forward_backend(self, x: torch.Tensor, backend: str) -> torch.Tensor:
        if backend == "dense_nvfp4":
            return self.forward_w4a4(x)
        if backend == "marlin_nvfp4":
            return self.forward_w4a16(x)
        raise ValueError(f"Unsupported shared NVFP4 backend: {backend}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = int(x.reshape(-1, self.in_features).size(0))
        backend = self.decode_backend if m <= self.marlin_m_threshold else self.prefill_backend
        return self.forward_backend(x, backend)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"marlin_m_threshold={self.marlin_m_threshold}, "
            f"prefill_backend={self.prefill_backend}, decode_backend={self.decode_backend}, "
            f"decode_activation_dtype={self.decode_activation_dtype}"
        )


class QwenManualHybridLinear(nn.Module):
    """Manual per-linear hybrid wrapper with separate prefill and decode backends."""

    in_features: int
    out_features: int

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        prefill_backend: str,
        decode_backend: str,
        decode_m_threshold: int,
        modules: dict[str, nn.Module],
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend
        self.decode_m_threshold = int(decode_m_threshold)
        self.backends = nn.ModuleDict(modules)
        self.eval()
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = int(x.reshape(-1, self.in_features).size(0))
        backend = self.decode_backend if m <= self.decode_m_threshold else self.prefill_backend
        return self.backends[backend](x)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"prefill_backend={self.prefill_backend}, decode_backend={self.decode_backend}, "
            f"decode_m_threshold={self.decode_m_threshold}"
        )


@dataclass(frozen=True)
class QwenKernelCheckpointBuildResult:
    metadata: dict[str, Any]
    state_dict: dict[str, torch.Tensor]
    report: Report


def prepare_qwen3_5_kernel_checkpoint_payload(
    model: nn.Module,
    *,
    method: str,
    variant: str,
    model_path: str,
    activation_dtype: torch.dtype = torch.bfloat16,
    policy_path: str | Path | None = None,
) -> QwenKernelCheckpointBuildResult:
    if method == "dense_nvfp4":
        report = replace_linear_with_cutlass_nvfp4(model, "qwen3_5", CutlassNVFP4Config())
        checkpoint_format = QWEN3_5_KERNEL_CHECKPOINT_FORMAT
    elif method == "sparse_bf16":
        report = replace_linear_with_cutlass_sparse_bf16(
            model,
            "qwen3_5",
            CutlassSparseBF16Config(prune=True),
        )
        checkpoint_format = QWEN3_5_KERNEL_CHECKPOINT_FORMAT
    elif method == "sparse_nvfp4":
        report = replace_linear_with_cutlass_sparse_nvfp4(
            model,
            "qwen3_5",
            CutlassSparseNVFP4Config(prune=True),
        )
        checkpoint_format = QWEN3_5_KERNEL_CHECKPOINT_FORMAT
    elif method == "marlin_nvfp4":
        metadata, report = prepare_marlin_nvfp4_packed_model(
            model,
            "qwen3_5",
            MarlinNVFP4Config(activation_dtype=activation_dtype),
        )
        metadata.update(
            {
                "model_family": "qwen3_5",
                "model_variant": variant,
                "model_path": model_path,
            }
        )
        return QwenKernelCheckpointBuildResult(
            metadata=metadata,
            state_dict=_cpu_state_dict(model),
            report=report,
        )
    elif method in QWEN3_5_HYBRID_NVFP4_METHODS:
        report = replace_linear_with_qwen_hybrid_nvfp4(
            model,
            hybrid_scheme=method,
            activation_dtype=activation_dtype,
        )
        checkpoint_format = QWEN3_5_HYBRID_NVFP4_CHECKPOINT_FORMAT
    elif method in QWEN3_5_MANUAL_HYBRID_METHODS:
        report = replace_linear_with_qwen_manual_hybrid(
            model,
            hybrid_scheme=method,
            activation_dtype=activation_dtype,
        )
        checkpoint_format = QWEN3_5_MANUAL_HYBRID_CHECKPOINT_FORMAT
    elif method in QWEN3_5_SWH_METHODS:
        report = replace_linear_with_qwen_swh(
            model,
            activation_dtype=activation_dtype,
        )
        checkpoint_format = QWEN3_5_SWH_CHECKPOINT_FORMAT
    elif method in QWEN3_5_PREDICTOR_HYBRID_METHODS:
        if policy_path is None:
            raise ValueError("policy_path is required for predictor_hybrid")
        report = replace_linear_with_qwen_predictor_hybrid(
            model,
            policy_path=policy_path,
            activation_dtype=activation_dtype,
        )
        checkpoint_format = QWEN3_5_PREDICTOR_HYBRID_CHECKPOINT_FORMAT
    else:
        raise ValueError(f"Unsupported Qwen3.5 kernel checkpoint method: {method}")

    metadata = {
        "checkpoint_format": checkpoint_format,
        "method": method,
        "model_family": "qwen3_5",
        "model_variant": variant,
        "model_path": model_path,
        "replacement_backend": report.backend,
        "replacement_config": report.config,
        "replaced_linear_count": report.replaced_linear_count,
        "skipped_linear_count": report.skipped_linear_count,
        "skipped": report.skipped,
        "module_specs": _module_specs_from_packed_model(model, method),
    }
    if policy_path is not None:
        metadata["policy_path"] = str(policy_path)
    return QwenKernelCheckpointBuildResult(metadata=metadata, state_dict=_cpu_state_dict(model), report=report)


def load_qwen3_5_kernel_checkpoint_into_model(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> tuple[dict[str, Any], Report]:
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"Invalid Qwen3.5 kernel checkpoint: {checkpoint_path}")
    metadata = dict(payload.get("metadata", {}))
    state_dict = payload["state_dict"]
    checkpoint_format = metadata.get("checkpoint_format")
    method = metadata.get("method")

    skip_strict_load = False
    if checkpoint_format == MARLIN_CHECKPOINT_FORMAT or method == "marlin_nvfp4":
        report = install_marlin_nvfp4_modules_from_state_dict(model, state_dict, metadata, device=device)
    elif checkpoint_format == QWEN3_5_HYBRID_NVFP4_CHECKPOINT_FORMAT or method == "hybrid_nvfp4":
        report = _install_hybrid_nvfp4(model, state_dict, metadata, device=device)
    elif checkpoint_format == QWEN3_5_MANUAL_HYBRID_CHECKPOINT_FORMAT or method in QWEN3_5_MANUAL_HYBRID_METHODS:
        report = _install_manual_hybrid(model, state_dict, metadata, device=device)
        skip_strict_load = True
    elif checkpoint_format == QWEN3_5_SWH_CHECKPOINT_FORMAT or method in QWEN3_5_SWH_METHODS:
        report = _install_manual_hybrid(model, state_dict, metadata, device=device)
        skip_strict_load = True
    elif checkpoint_format == QWEN3_5_PREDICTOR_HYBRID_CHECKPOINT_FORMAT or method in QWEN3_5_PREDICTOR_HYBRID_METHODS:
        report = _install_manual_hybrid(model, state_dict, metadata, device=device)
        skip_strict_load = True
    elif checkpoint_format == QWEN3_5_KERNEL_CHECKPOINT_FORMAT:
        report = _install_cutlass_modules_from_state_dict(model, state_dict, metadata, device=device)
    else:
        raise ValueError(
            f"Unsupported checkpoint_format={checkpoint_format}; "
            f"expected {QWEN3_5_KERNEL_CHECKPOINT_FORMAT}, "
            f"{QWEN3_5_HYBRID_NVFP4_CHECKPOINT_FORMAT}, "
            f"{QWEN3_5_MANUAL_HYBRID_CHECKPOINT_FORMAT}, or {MARLIN_CHECKPOINT_FORMAT}"
        )

    if not skip_strict_load:
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Failed to load Qwen3.5 kernel checkpoint: missing={missing}, unexpected={unexpected}")
    if str(device) != "auto":
        model.to(device)
    model.eval()
    metadata["checkpoint_path"] = str(checkpoint_path)
    metadata["packed_checkpoint_file_size_bytes"] = checkpoint_path.stat().st_size
    return metadata, report


def default_qwen3_5_kernel_checkpoint_path(variant: str, method: str) -> str:
    variant_key = variant.lower().replace(".", "_")
    return f"artifacts/checkpoints/qwen3_5_{variant_key}/{method}/model.pt"


def replace_linear_with_qwen_hybrid_nvfp4(
    model: nn.Module,
    *,
    hybrid_scheme: str = "hybrid_nvfp4",
    activation_dtype: torch.dtype = torch.bfloat16,
    marlin_m_threshold: int = 16,
) -> QwenHybridNVFP4ReplacementReport:
    from fake.compression.modules import select_compressible_modules

    if hybrid_scheme not in QWEN3_5_HYBRID_NVFP4_METHODS:
        raise ValueError(f"Unsupported Qwen3.5 hybrid NVFP4 scheme: {hybrid_scheme}")

    wrapper = _load_wrapper()
    skipped: list[dict[str, str]] = []
    replaced = 0
    selected = select_compressible_modules(model, "qwen3_5")
    targets = [(info.name, info.kind) for info in selected]
    del selected
    for module_name, kind in targets:
        if kind != "linear":
            skipped.append({"name": module_name, "reason": f"unsupported_kind:{kind}"})
            continue
        parent = model
        parts = module_name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        child_name = parts[-1]
        linear = getattr(parent, child_name)
        if not isinstance(linear, nn.Linear):
            skipped.append({"name": module_name, "reason": f"not_linear:{type(linear).__name__}"})
            continue
        if hybrid_scheme == "hybrid_nvfp4_major" and not _is_major_dense_nvfp4_hybrid_target(module_name):
            skipped.append({"name": module_name, "reason": "policy_keep_bf16"})
            continue
        if not wrapper.can_use_cutlass_nvfp4(
            1,
            linear.out_features,
            linear.in_features,
            load_extension=False,
        ):
            skipped.append({"name": module_name, "reason": _hybrid_shape_reason(linear, "dense_nvfp4")})
            continue
        if not wrapper.can_use_marlin_nvfp4(
            1,
            linear.out_features,
            linear.in_features,
            load_extension=False,
        ):
            skipped.append({"name": module_name, "reason": _hybrid_shape_reason(linear, "marlin_nvfp4")})
            continue
        canonical = wrapper.canonical_from_linear(linear)
        setattr(
            parent,
            child_name,
            QwenHybridDenseNVFP4Linear(
                canonical,
                decode_activation_dtype=activation_dtype,
                marlin_m_threshold=marlin_m_threshold,
            ),
        )
        replaced += 1
    return QwenHybridNVFP4ReplacementReport(
        backend=f"qwen_{hybrid_scheme}_dense_nvfp4_cutlass_w4a4_marlin_w4a16",
        config={
            "decode_activation_dtype": str(activation_dtype),
            "marlin_m_threshold": marlin_m_threshold,
            "hybrid_scheme": hybrid_scheme,
            "same_weight_scope": "dense_nvfp4_only",
        },
        replaced_linear_count=replaced,
        skipped_linear_count=len(skipped),
        skipped=skipped,
    )


def _install_hybrid_nvfp4(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    *,
    device: str | torch.device = "cuda",
) -> QwenHybridNVFP4ReplacementReport:
    wrapper = _load_wrapper()
    replacement_config = dict(metadata.get("replacement_config", {}))
    activation_dtype = _dtype_from_string(replacement_config.get("decode_activation_dtype", "torch.bfloat16"))
    marlin_m_threshold = int(replacement_config.get("marlin_m_threshold", 16))
    replaced = 0
    skipped: list[dict[str, str]] = []
    for spec in metadata.get("module_specs", []):
        name = spec["name"]
        try:
            target_device = _module_target_device(model, name, device)
            canonical = wrapper.NVFP4CanonicalWeight(
                packed_weight=state_dict[f"{name}.canonical_packed_weight"].to(target_device),
                logical_scale=state_dict[f"{name}.canonical_logical_scale"].to(target_device),
                global_scale=state_dict[f"{name}.canonical_global_scale"].to(target_device),
                in_features=int(spec["in_features"]),
                out_features=int(spec["out_features"]),
                original_dtype=_dtype_from_string(spec.get("original_dtype", "torch.bfloat16")),
                bias=_optional_tensor_to(state_dict.get(f"{name}.bias"), target_device),
            )
            _set_module(
                model,
                name,
                QwenHybridDenseNVFP4Linear(
                    canonical,
                    decode_activation_dtype=activation_dtype,
                    marlin_m_threshold=marlin_m_threshold,
                ),
            )
            replaced += 1
        except Exception as exc:
            skipped.append({"name": name, "reason": f"{type(exc).__name__}:{exc}"})
    return QwenHybridNVFP4ReplacementReport(
        backend=str(metadata.get("replacement_backend", "qwen_hybrid_dense_nvfp4_cutlass_w4a4_marlin_w4a16")),
        config=replacement_config,
        replaced_linear_count=replaced,
        skipped_linear_count=int(metadata.get("skipped_linear_count", 0)) + len(skipped),
        skipped=list(metadata.get("skipped", [])) + skipped,
    )


def replace_linear_with_qwen_manual_hybrid(
    model: nn.Module,
    *,
    hybrid_scheme: str,
    activation_dtype: torch.dtype = torch.bfloat16,
) -> QwenManualHybridReplacementReport:
    from fake.compression.modules import select_compressible_modules

    if hybrid_scheme not in QWEN3_5_MANUAL_HYBRID_METHODS:
        raise ValueError(f"Unsupported Qwen3.5 manual hybrid scheme: {hybrid_scheme}")

    skipped: list[dict[str, str]] = []
    backend_counts: dict[str, int] = {}
    replaced = 0
    decode_m_threshold = _manual_decode_m_threshold(hybrid_scheme)
    selected = select_compressible_modules(model, "qwen3_5")
    targets = [(info.name, info.kind) for info in selected]
    del selected
    for module_name, kind in targets:
        if kind != "linear":
            skipped.append({"name": module_name, "reason": f"unsupported_kind:{kind}"})
            continue
        parent, child_name = _resolve_parent(model, module_name)
        linear = getattr(parent, child_name)
        if not isinstance(linear, nn.Linear):
            skipped.append({"name": module_name, "reason": f"not_linear:{type(linear).__name__}"})
            continue

        prefill_backend = _manual_prefill_backend(module_name, hybrid_scheme)
        decode_backend = _manual_decode_backend(module_name, hybrid_scheme)
        needed = tuple(dict.fromkeys((prefill_backend, decode_backend)))
        modules: dict[str, nn.Module] = {}
        module_skipped = False
        for backend in needed:
            try:
                modules[backend] = _build_manual_backend_module(
                    linear,
                    backend,
                    activation_dtype=activation_dtype,
                )
                backend_counts[backend] = backend_counts.get(backend, 0) + 1
            except Exception as exc:
                skipped.append({"name": module_name, "reason": f"{backend}:{type(exc).__name__}:{exc}"})
                module_skipped = True
                break
        if module_skipped:
            continue
        setattr(
            parent,
            child_name,
            QwenManualHybridLinear(
                in_features=linear.in_features,
                out_features=linear.out_features,
                prefill_backend=prefill_backend,
                decode_backend=decode_backend,
                decode_m_threshold=decode_m_threshold,
                modules=modules,
            ),
        )
        replaced += 1

    return QwenManualHybridReplacementReport(
        hybrid_scheme=hybrid_scheme,
        config={
            "decode_activation_dtype": str(activation_dtype),
            "decode_m_threshold": decode_m_threshold,
            "manual_policy": "qwen3_5_2b_kernel_benchmark_suffix_map",
        },
        replaced_linear_count=replaced,
        skipped_linear_count=len(skipped),
        skipped=skipped,
        backend_counts=backend_counts,
    )


def replace_linear_with_qwen_weight_sharing_hybrid(
    model: nn.Module,
    *,
    activation_dtype: torch.dtype = torch.bfloat16,
    marlin_m_threshold: int = 64,
) -> QwenManualHybridReplacementReport:
    """Weight-sharing hybrid: single canonical NVFP4 weight, dual backend.

    Policy:
      - in_proj_a/b:  sparse_bf16 prefill + bf16 decode (N=32, too small for marlin)
      - k_proj, v_proj: marlin only (both phases use marlin_nvfp4)
      - everything else: dense_nvfp4 prefill + marlin_nvfp4 decode
        via QwenHybridDenseNVFP4Linear (shared canonical NVFP4 weight)
    """
    from fake.compression.modules import select_compressible_modules

    GROUP_A = {'linear_attn.in_proj_a', 'linear_attn.in_proj_b'}
    GROUP_B = {'self_attn.k_proj', 'self_attn.v_proj'}

    skipped: list[dict[str, str]] = []
    backend_counts: dict[str, int] = {}
    replaced = 0
    wrapper = _load_wrapper()
    selected = select_compressible_modules(model, "qwen3_5")
    targets = [(info.name, info.kind) for info in selected]
    del selected

    for module_name, kind in targets:
        if kind != "linear":
            skipped.append({"name": module_name, "reason": f"unsupported_kind:{kind}"})
            continue
        parent, child_name = _resolve_parent(model, module_name)
        linear = getattr(parent, child_name)
        if not isinstance(linear, nn.Linear):
            skipped.append({"name": module_name, "reason": f"not_linear:{type(linear).__name__}"})
            continue

        try:
            if any(module_name.endswith(g) for g in GROUP_A):
                pf_mod = _build_manual_backend_module(linear, "sparse_bf16", activation_dtype=activation_dtype)
                dec_mod = _build_manual_backend_module(linear, "bf16", activation_dtype=activation_dtype)
                setattr(parent, child_name, QwenManualHybridLinear(
                    in_features=linear.in_features,
                    out_features=linear.out_features,
                    prefill_backend="sparse_bf16",
                    decode_backend="bf16",
                    decode_m_threshold=marlin_m_threshold,
                    modules={"sparse_bf16": pf_mod, "bf16": dec_mod},
                ))
                backend_counts["manual(sparse_bf16/bf16)"] = backend_counts.get("manual(sparse_bf16/bf16)", 0) + 1
            elif any(module_name.endswith(g) for g in GROUP_B):
                setattr(parent, child_name,
                    wrapper.MarlinNVFP4Linear.from_linear(linear, activation_dtype=activation_dtype))
                backend_counts["marlin_only"] = backend_counts.get("marlin_only", 0) + 1
            else:
                canonical = wrapper.canonical_from_linear(linear, device=linear.weight.device)
                setattr(parent, child_name, QwenHybridDenseNVFP4Linear(
                    canonical,
                    decode_activation_dtype=activation_dtype,
                    marlin_m_threshold=marlin_m_threshold,
                ))
                backend_counts["hybrid_dense_nvfp4(marlin)"] = backend_counts.get("hybrid_dense_nvfp4(marlin)", 0) + 1
            replaced += 1
        except Exception as exc:
            skipped.append({"name": module_name, "reason": f"{type(exc).__name__}:{exc}"})

    return QwenManualHybridReplacementReport(
        hybrid_scheme="weight_sharing_hybrid",
        config={
            "decode_activation_dtype": str(activation_dtype),
            "marlin_m_threshold": marlin_m_threshold,
            "policy": "weight_sharing_dense_nvfp4_marlin_with_sparse_bf16_on_tiny_layers",
        },
        replaced_linear_count=replaced,
        skipped_linear_count=len(skipped),
        skipped=skipped,
        backend_counts=backend_counts,
    )


def replace_linear_with_qwen_swh(
    model: nn.Module,
    *,
    activation_dtype: torch.dtype = torch.bfloat16,
) -> QwenManualHybridReplacementReport:
    """Replace model linears with shape-workload hybrid using data-driven policy.

    Uses the optimal kernel per-layer based on measured benchmark data
    for Qwen3.5-9B. Prefill uses sparse_nvfp4/sparse_bf16/marlin_nvfp4;
    decode uses marlin_nvfp4 (with dense_bf16 on tiny N=32 layers).
    """
    from fake.compression.modules import select_compressible_modules

    skipped: list[dict[str, str]] = []
    backend_counts: dict[str, int] = {}
    replaced = 0
    decode_m_threshold = 64
    selected = select_compressible_modules(model, "qwen3_5")
    targets = [(info.name, info.kind) for info in selected]
    del selected
    for module_name, kind in targets:
        if kind != "linear":
            skipped.append({"name": module_name, "reason": f"unsupported_kind:{kind}"})
            continue
        parent, child_name = _resolve_parent(model, module_name)
        linear = getattr(parent, child_name)
        if not isinstance(linear, nn.Linear):
            skipped.append({"name": module_name, "reason": f"not_linear:{type(linear).__name__}"})
            continue

        prefill_backend = _swh_prefill_backend(module_name)
        decode_backend = _swh_decode_backend(module_name)
        needed = tuple(dict.fromkeys((prefill_backend, decode_backend)))
        modules: dict[str, nn.Module] = {}
        module_skipped = False
        for backend in needed:
            try:
                modules[backend] = _build_manual_backend_module(
                    linear, backend, activation_dtype=activation_dtype,
                )
                backend_counts[backend] = backend_counts.get(backend, 0) + 1
            except Exception as exc:
                skipped.append({"name": module_name, "reason": f"{backend}:{type(exc).__name__}:{exc}"})
                module_skipped = True
                break
        if module_skipped:
            continue
        setattr(
            parent,
            child_name,
            QwenManualHybridLinear(
                in_features=linear.in_features,
                out_features=linear.out_features,
                prefill_backend=prefill_backend,
                decode_backend=decode_backend,
                decode_m_threshold=decode_m_threshold,
                modules=modules,
            ),
        )
        replaced += 1

    return QwenManualHybridReplacementReport(
        hybrid_scheme="shape_workload_hybrid",
        config={
            "decode_activation_dtype": str(activation_dtype),
            "decode_m_threshold": decode_m_threshold,
            "policy": "data_driven_qwen35_9b_module_kernel_benchmarks",
        },
        replaced_linear_count=replaced,
        skipped_linear_count=len(skipped),
        skipped=skipped,
        backend_counts=backend_counts,
    )


def replace_linear_with_qwen_predictor_hybrid(
    model: nn.Module,
    *,
    policy_path: str | Path,
    activation_dtype: torch.dtype = torch.bfloat16,
) -> QwenManualHybridReplacementReport:
    """Replace Qwen3.5 linears using a generic offline hybrid policy."""
    from fake.compression.modules import select_compressible_modules

    policy = load_policy_json(policy_path)
    decode_m_threshold = int(policy.scenario.get("m_decode", policy.scenario.get("batch_size", 1)))
    skipped: list[dict[str, str]] = []
    backend_counts: dict[str, int] = {}
    replaced = 0
    wrapper = _load_wrapper()

    selected = select_compressible_modules(model, "qwen3_5")
    targets = [(info.name, info.kind) for info in selected]
    del selected
    for module_name, kind in targets:
        if kind != "linear":
            skipped.append({"name": module_name, "reason": f"unsupported_kind:{kind}"})
            continue
        decision = _policy_decision_for_module(policy, module_name)
        if decision is None:
            skipped.append({"name": module_name, "reason": "missing_policy"})
            continue
        if decision.selected_prefill_backend is None or decision.selected_decode_backend is None:
            skipped.append({"name": module_name, "reason": decision.reason or "unselected_policy"})
            continue

        parent, child_name = _resolve_parent(model, module_name)
        linear = getattr(parent, child_name)
        if not isinstance(linear, nn.Linear):
            skipped.append({"name": module_name, "reason": f"not_linear:{type(linear).__name__}"})
            continue
        if int(decision.n) != int(linear.out_features) or int(decision.k) != int(linear.in_features):
            skipped.append(
                {
                    "name": module_name,
                    "reason": (
                        f"policy_shape_mismatch:policy=({decision.n},{decision.k}),"
                        f"model=({linear.out_features},{linear.in_features})"
                    ),
                }
            )
            continue

        prefill_backend = _policy_backend_to_manual(decision.selected_prefill_backend)
        decode_backend = _policy_backend_to_manual(decision.selected_decode_backend)
        if prefill_backend == decode_backend == "bf16":
            backend_counts["bf16"] = backend_counts.get("bf16", 0) + 1
            continue
        try:
            if _is_shared_nvfp4_policy(prefill_backend, decode_backend):
                canonical = wrapper.canonical_from_linear(linear, device=linear.weight.device)
                setattr(
                    parent,
                    child_name,
                    QwenHybridDenseNVFP4Linear(
                        canonical,
                        decode_activation_dtype=activation_dtype,
                        marlin_m_threshold=decode_m_threshold,
                        prefill_backend=prefill_backend,
                        decode_backend=decode_backend,
                    ),
                )
                backend_counts[f"{prefill_backend}/{decode_backend}"] = backend_counts.get(f"{prefill_backend}/{decode_backend}", 0) + 1
            else:
                needed = tuple(dict.fromkeys((prefill_backend, decode_backend)))
                modules = {
                    backend: _build_manual_backend_module(linear, backend, activation_dtype=activation_dtype)
                    for backend in needed
                }
                setattr(
                    parent,
                    child_name,
                    QwenManualHybridLinear(
                        in_features=linear.in_features,
                        out_features=linear.out_features,
                        prefill_backend=prefill_backend,
                        decode_backend=decode_backend,
                        decode_m_threshold=decode_m_threshold,
                        modules=modules,
                    ),
                )
                for backend in needed:
                    backend_counts[backend] = backend_counts.get(backend, 0) + 1
            replaced += 1
        except Exception as exc:
            skipped.append({"name": module_name, "reason": f"{type(exc).__name__}:{exc}"})

    return QwenManualHybridReplacementReport(
        hybrid_scheme="predictor_hybrid",
        config={
            "decode_activation_dtype": str(activation_dtype),
            "decode_m_threshold": decode_m_threshold,
            "policy": "offline_hybrid_policy",
            "policy_path": str(policy_path),
            "policy_format": policy.policy_format,
            "include_conversion_cost": policy.include_conversion_cost,
            "scenario": policy.scenario,
        },
        replaced_linear_count=replaced,
        skipped_linear_count=len(skipped),
        skipped=skipped,
        backend_counts=backend_counts,
    )


def _policy_decision_for_module(policy: Any, module_name: str) -> Any | None:
    for decision in policy.modules:
        if module_name == decision.name or module_name.endswith(f".{decision.name}"):
            return decision
    return None


def _install_manual_hybrid(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    *,
    device: str | torch.device = "cuda",
) -> QwenManualHybridReplacementReport:
    replacement_config = dict(metadata.get("replacement_config", {}))
    activation_dtype = _dtype_from_string(replacement_config.get("decode_activation_dtype", "torch.bfloat16"))
    decode_m_threshold = int(replacement_config.get("decode_m_threshold", 16))
    hybrid_scheme = str(metadata.get("method", "manual_hybrid"))
    skipped: list[dict[str, str]] = []
    backend_counts: dict[str, int] = {}
    replaced = 0
    for spec in metadata.get("module_specs", []):
        name = spec["name"]
        prefill_backend = spec["prefill_backend"]
        decode_backend = spec["decode_backend"]
        needed = tuple(dict.fromkeys((prefill_backend, decode_backend)))
        try:
            target_device = _module_target_device(model, name, device)
            if spec.get("shared_nvfp4"):
                wrapper = _load_wrapper()
                canonical = wrapper.NVFP4CanonicalWeight(
                    packed_weight=state_dict[f"{name}.canonical_packed_weight"].to(target_device),
                    logical_scale=state_dict[f"{name}.canonical_logical_scale"].to(target_device),
                    global_scale=state_dict[f"{name}.canonical_global_scale"].to(target_device),
                    in_features=int(spec["in_features"]),
                    out_features=int(spec["out_features"]),
                    original_dtype=_dtype_from_string(spec.get("original_dtype", "torch.bfloat16")),
                    bias=_optional_tensor_to(state_dict.get(f"{name}.bias"), target_device),
                )
                _set_module(
                    model,
                    name,
                    QwenHybridDenseNVFP4Linear(
                        canonical,
                        decode_activation_dtype=activation_dtype,
                        marlin_m_threshold=decode_m_threshold,
                        prefill_backend=prefill_backend,
                        decode_backend=decode_backend,
                    ),
                )
            else:
                modules = {
                    backend: _manual_backend_from_state(
                        state_dict,
                        name,
                        backend,
                        spec,
                        target_device,
                        activation_dtype=activation_dtype,
                    )
                    for backend in needed
                }
                _set_module(
                    model,
                    name,
                    QwenManualHybridLinear(
                        in_features=int(spec["in_features"]),
                        out_features=int(spec["out_features"]),
                        prefill_backend=prefill_backend,
                        decode_backend=decode_backend,
                        decode_m_threshold=decode_m_threshold,
                        modules=modules,
                    ),
                )
            for backend in needed:
                backend_counts[backend] = backend_counts.get(backend, 0) + 1
            replaced += 1
        except Exception as exc:
            skipped.append({"name": name, "reason": f"{type(exc).__name__}:{exc}"})
    return QwenManualHybridReplacementReport(
        hybrid_scheme=hybrid_scheme,
        config=replacement_config,
        replaced_linear_count=replaced,
        skipped_linear_count=int(metadata.get("skipped_linear_count", 0)) + len(skipped),
        skipped=list(metadata.get("skipped", [])) + skipped,
        backend_counts=backend_counts,
    )


def _install_cutlass_modules_from_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    *,
    device: str | torch.device = "cuda",
) -> ReplacementReport | SparseBF16ReplacementReport | SparseReplacementReport:
    method = metadata.get("method")
    if method == "dense_nvfp4":
        return _install_dense_nvfp4(model, state_dict, metadata, device=device)
    if method == "sparse_bf16":
        return _install_sparse_bf16(model, state_dict, metadata, device=device)
    if method == "sparse_nvfp4":
        return _install_sparse_nvfp4(model, state_dict, metadata, device=device)
    raise ValueError(f"Unsupported Qwen3.5 CUTLASS method: {method}")


def _install_dense_nvfp4(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    *,
    device: str | torch.device = "cuda",
) -> ReplacementReport:
    wrapper = _load_wrapper()
    replaced = 0
    skipped: list[dict[str, str]] = []
    for spec in metadata.get("module_specs", []):
        name = spec["name"]
        try:
            target_device = _module_target_device(model, name, device)
            weight = wrapper.NVFP4Weight(
                packed_weight=state_dict[f"{name}.packed_weight"].to(target_device),
                scale=state_dict[f"{name}.weight_scale"].to(target_device),
                global_scale=state_dict[f"{name}.weight_global_scale"].to(target_device),
                in_features=int(spec["in_features"]),
                out_features=int(spec["out_features"]),
                original_dtype=_dtype_from_string(spec.get("original_dtype", "torch.bfloat16")),
                bias=_optional_tensor_to(state_dict.get(f"{name}.bias"), target_device),
            )
            _set_module(model, name, wrapper.NVFP4Linear(weight))
            replaced += 1
        except Exception as exc:
            skipped.append({"name": name, "reason": f"{type(exc).__name__}:{exc}"})
    return ReplacementReport(
        backend="cutlass_nvfp4_sm120",
        config=dict(metadata.get("replacement_config", {})),
        replaced_linear_count=replaced,
        skipped_linear_count=int(metadata.get("skipped_linear_count", 0)) + len(skipped),
        skipped=list(metadata.get("skipped", [])) + skipped,
    )


def _install_sparse_bf16(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    *,
    device: str | torch.device = "cuda",
) -> SparseBF16ReplacementReport:
    wrapper = _load_wrapper()
    pad_multiple = int(metadata.get("token_pad_multiple", metadata.get("replacement_config", {}).get("pad_tokens_to_multiple", 8)))
    replaced = 0
    skipped: list[dict[str, str]] = []
    for spec in metadata.get("module_specs", []):
        name = spec["name"]
        prefix = f"{name}.sparse_linear"
        try:
            target_device = _module_target_device(model, name, device)
            weight = wrapper.SparseBF16Weight(
                sparse_weight=state_dict[f"{prefix}.sparse_weight"].to(target_device),
                metadata=state_dict[f"{prefix}.metadata"].to(target_device),
                in_features=int(spec["in_features"]),
                out_features=int(spec["out_features"]),
                original_dtype=_dtype_from_string(spec.get("original_dtype", "torch.bfloat16")),
                bias=_optional_tensor_to(state_dict.get(f"{prefix}.bias"), target_device),
            )
            _set_module(model, name, PaddedSparseBF16Linear(wrapper.SparseBF16Linear(weight), pad_multiple))
            replaced += 1
        except Exception as exc:
            skipped.append({"name": name, "reason": f"{type(exc).__name__}:{exc}"})
    return SparseBF16ReplacementReport(
        backend="cutlass_sparse_bf16_cusparselt",
        config=dict(metadata.get("replacement_config", {})),
        replaced_linear_count=replaced,
        skipped_linear_count=int(metadata.get("skipped_linear_count", 0)) + len(skipped),
        skipped=list(metadata.get("skipped", [])) + skipped,
    )


def _install_sparse_nvfp4(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    *,
    device: str | torch.device = "cuda",
) -> SparseReplacementReport:
    wrapper = _load_wrapper()
    pad_multiple = int(metadata.get("token_pad_multiple", metadata.get("replacement_config", {}).get("pad_tokens_to_multiple", 32)))
    replaced = 0
    skipped: list[dict[str, str]] = []
    for spec in metadata.get("module_specs", []):
        name = spec["name"]
        prefix = f"{name}.sparse_linear"
        try:
            target_device = _module_target_device(model, name, device)
            weight = wrapper.SparseNVFP4Weight(
                sparse_weight=state_dict[f"{prefix}.sparse_weight"].to(target_device),
                metadata=state_dict[f"{prefix}.metadata"].to(target_device),
                scale=state_dict[f"{prefix}.weight_scale"].to(target_device),
                global_scale=state_dict[f"{prefix}.weight_global_scale"].to(target_device),
                in_features=int(spec["in_features"]),
                out_features=int(spec["out_features"]),
                original_dtype=_dtype_from_string(spec.get("original_dtype", "torch.bfloat16")),
                bias=_optional_tensor_to(state_dict.get(f"{prefix}.bias"), target_device),
            )
            _set_module(model, name, PaddedSparseNVFP4Linear(wrapper.SparseNVFP4Linear(weight), pad_multiple))
            replaced += 1
        except Exception as exc:
            skipped.append({"name": name, "reason": f"{type(exc).__name__}:{exc}"})
    return SparseReplacementReport(
        backend="cutlass_sparse_nvfp4_sm120",
        config=dict(metadata.get("replacement_config", {})),
        replaced_linear_count=replaced,
        skipped_linear_count=int(metadata.get("skipped_linear_count", 0)) + len(skipped),
        skipped=list(metadata.get("skipped", [])) + skipped,
    )


def _module_specs_from_packed_model(model: nn.Module, method: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    nvfp4_linear_cls, _ = _load_cutlass_nvfp4_symbols()
    for name, module in model.named_modules():
        if method == "dense_nvfp4" and isinstance(module, nvfp4_linear_cls):
            specs.append(_packed_spec(name, module))
        elif method in QWEN3_5_HYBRID_NVFP4_METHODS and isinstance(module, QwenHybridDenseNVFP4Linear):
            specs.append(_packed_spec(name, module))
        elif method in QWEN3_5_PREDICTOR_HYBRID_METHODS and isinstance(module, QwenHybridDenseNVFP4Linear):
            spec = _packed_spec(name, module)
            spec["prefill_backend"] = module.prefill_backend
            spec["decode_backend"] = module.decode_backend
            spec["shared_nvfp4"] = True
            specs.append(spec)
        elif (
            method in QWEN3_5_MANUAL_HYBRID_METHODS
            or method in QWEN3_5_SWH_METHODS
            or method in QWEN3_5_PREDICTOR_HYBRID_METHODS
        ) and isinstance(module, QwenManualHybridLinear):
            spec = _packed_spec(name, module)
            spec["prefill_backend"] = module.prefill_backend
            spec["decode_backend"] = module.decode_backend
            specs.append(spec)
        elif method == "sparse_bf16" and isinstance(module, PaddedSparseBF16Linear):
            specs.append(_packed_spec(name, module.sparse_linear))
        elif method == "sparse_nvfp4" and isinstance(module, PaddedSparseNVFP4Linear):
            specs.append(_packed_spec(name, module.sparse_linear))
    return specs


def _packed_spec(name: str, module: nn.Module) -> dict[str, Any]:
    return {
        "name": name,
        "in_features": int(module.in_features),
        "out_features": int(module.out_features),
        "bias": getattr(module, "bias", None) is not None,
        "original_dtype": str(getattr(module, "original_dtype", torch.bfloat16)),
    }


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _resolve_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parent = model
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _set_module(model: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent, child_name = _resolve_parent(model, module_name)
    setattr(parent, child_name, new_module)


def _get_module(model: nn.Module, module_name: str) -> nn.Module:
    module = model
    for part in module_name.split("."):
        module = getattr(module, part)
    return module


def _module_target_device(
    model: nn.Module,
    module_name: str,
    device: str | torch.device,
) -> torch.device:
    if str(device) != "auto":
        return torch.device(device)
    module = _get_module(model, module_name)
    for tensor in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
        return tensor.device
    for tensor in list(module.parameters()) + list(module.buffers()):
        return tensor.device
    return torch.device("cuda")


def _optional_tensor_to(tensor: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    return None if tensor is None else tensor.to(device)


def _hybrid_shape_reason(linear: nn.Linear, backend: str) -> str:
    return f"shape_not_supported:{backend}:in_features={linear.in_features},out_features={linear.out_features}"


def _is_major_dense_nvfp4_hybrid_target(module_name: str) -> bool:
    return module_name.endswith(
        (
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z",
            "linear_attn.out_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "self_attn.q_proj",
            "self_attn.o_proj",
        )
    )


def _build_manual_backend_module(
    linear: nn.Linear,
    backend: str,
    *,
    activation_dtype: torch.dtype,
) -> nn.Module:
    wrapper = _load_wrapper()
    backend = _policy_backend_to_manual(backend)
    if backend == "bf16":
        dense = nn.Linear(linear.in_features, linear.out_features, bias=linear.bias is not None)
        dense = dense.to(device=linear.weight.device, dtype=torch.bfloat16)
        dense.weight.data.copy_(linear.weight.detach().to(torch.bfloat16))
        if linear.bias is not None:
            dense.bias.data.copy_(linear.bias.detach().to(torch.bfloat16))
        dense.eval()
        dense.requires_grad_(False)
        return dense
    if backend == "dense_nvfp4":
        if not wrapper.can_use_cutlass_nvfp4(1, linear.out_features, linear.in_features, load_extension=False):
            raise ValueError(_hybrid_shape_reason(linear, backend))
        return wrapper.NVFP4Linear.from_linear(linear)
    if backend == "marlin_nvfp4":
        if not wrapper.can_use_marlin_nvfp4(1, linear.out_features, linear.in_features, load_extension=False):
            raise ValueError(_hybrid_shape_reason(linear, backend))
        return wrapper.MarlinNVFP4Linear.from_linear(linear, activation_dtype=activation_dtype)
    if backend == "sparse_bf16":
        if (linear.out_features, linear.in_features) in SPARSE_BF16_BLOCKED_SHAPES:
            raise ValueError(_hybrid_shape_reason(linear, "sparse_bf16_blocked"))
        sparse_cls, can_use = _load_cutlass_sparse_bf16_symbols()
        if not can_use(linear.out_features, 8, linear.in_features, load_extension=False):
            raise ValueError(_hybrid_shape_reason(linear, backend))
        return PaddedSparseBF16Linear(sparse_cls.from_linear(linear, prune=True), 8)
    if backend == "sparse_nvfp4":
        sparse_cls, can_use = _load_cutlass_sparse_nvfp4_symbols()
        if not can_use(linear.out_features, 32, linear.in_features, load_extension=False):
            raise ValueError(_hybrid_shape_reason(linear, backend))
        return PaddedSparseNVFP4Linear(sparse_cls.from_linear(linear, prune=True), 32)
    raise ValueError(f"Unsupported manual backend: {backend}")


def _manual_backend_from_state(
    state_dict: dict[str, torch.Tensor],
    module_name: str,
    backend: str,
    spec: dict[str, Any],
    device: torch.device,
    *,
    activation_dtype: torch.dtype,
) -> nn.Module:
    wrapper = _load_wrapper()
    backend = _policy_backend_to_manual(backend)
    prefix = f"{module_name}.backends.{backend}"
    in_features = int(spec["in_features"])
    out_features = int(spec["out_features"])
    original_dtype = _dtype_from_string(spec.get("original_dtype", "torch.bfloat16"))
    if backend == "bf16":
        dense = nn.Linear(in_features, out_features, bias=f"{prefix}.bias" in state_dict)
        dense = dense.to(device=device, dtype=torch.bfloat16)
        dense.weight.data.copy_(state_dict[f"{prefix}.weight"].to(device=device, dtype=torch.bfloat16))
        if dense.bias is not None:
            dense.bias.data.copy_(state_dict[f"{prefix}.bias"].to(device=device, dtype=torch.bfloat16))
        dense.eval()
        dense.requires_grad_(False)
        return dense
    if backend == "dense_nvfp4":
        weight = wrapper.NVFP4Weight(
            packed_weight=state_dict[f"{prefix}.packed_weight"].to(device),
            scale=state_dict[f"{prefix}.weight_scale"].to(device),
            global_scale=state_dict[f"{prefix}.weight_global_scale"].to(device),
            in_features=in_features,
            out_features=out_features,
            original_dtype=original_dtype,
            bias=_optional_tensor_to(state_dict.get(f"{prefix}.bias"), device),
        )
        return wrapper.NVFP4Linear(weight)
    if backend == "marlin_nvfp4":
        weight = wrapper.MarlinNVFP4Weight(
            packed_weight=state_dict[f"{prefix}.packed_weight"].to(device),
            weight_scale=state_dict[f"{prefix}.weight_scale"].to(device),
            global_scale=state_dict[f"{prefix}.weight_global_scale"].to(device),
            workspace=_make_marlin_workspace(device),
            in_features=in_features,
            out_features=out_features,
            activation_dtype=activation_dtype,
            original_dtype=original_dtype,
            bias=_optional_tensor_to(state_dict.get(f"{prefix}.bias"), device),
        )
        return wrapper.MarlinNVFP4Linear(weight)
    if backend == "sparse_bf16":
        sparse_prefix = f"{prefix}.sparse_linear"
        weight = wrapper.SparseBF16Weight(
            sparse_weight=state_dict[f"{sparse_prefix}.sparse_weight"].to(device),
            metadata=state_dict[f"{sparse_prefix}.metadata"].to(device),
            in_features=in_features,
            out_features=out_features,
            original_dtype=original_dtype,
            bias=_optional_tensor_to(state_dict.get(f"{sparse_prefix}.bias"), device),
        )
        return PaddedSparseBF16Linear(wrapper.SparseBF16Linear(weight), 8)
    if backend == "sparse_nvfp4":
        sparse_prefix = f"{prefix}.sparse_linear"
        weight = wrapper.SparseNVFP4Weight(
            sparse_weight=state_dict[f"{sparse_prefix}.sparse_weight"].to(device),
            metadata=state_dict[f"{sparse_prefix}.metadata"].to(device),
            scale=state_dict[f"{sparse_prefix}.weight_scale"].to(device),
            global_scale=state_dict[f"{sparse_prefix}.weight_global_scale"].to(device),
            in_features=in_features,
            out_features=out_features,
            original_dtype=original_dtype,
            bias=_optional_tensor_to(state_dict.get(f"{sparse_prefix}.bias"), device),
        )
        return PaddedSparseNVFP4Linear(wrapper.SparseNVFP4Linear(weight), 32)
    raise ValueError(f"Unsupported manual backend: {backend}")


def _policy_backend_to_manual(backend: str) -> str:
    if backend == "dense_bf16":
        return "bf16"
    return backend


def _is_shared_nvfp4_policy(prefill_backend: str, decode_backend: str) -> bool:
    return {prefill_backend, decode_backend} == {"dense_nvfp4", "marlin_nvfp4"}


def _manual_decode_m_threshold(hybrid_scheme: str) -> int:
    if hybrid_scheme == "manual_hybrid_m1":
        return 1
    if hybrid_scheme == "manual_hybrid_m4":
        return 4
    if hybrid_scheme == "manual_hybrid_m8":
        return 8
    if hybrid_scheme == "manual_hybrid_m16":
        return 16
    raise ValueError(f"Unsupported Qwen3.5 manual hybrid scheme: {hybrid_scheme}")


def _swh_prefill_backend(module_name: str) -> str:
    """Data-driven optimal prefill backend for Qwen3.5-9B at large M (>= 16384).

    Derived from measured module-level kernel benchmarks:
    artifacts/results/benchmarks/module/Qwen3.5-9B/kernel/
    """
    if module_name.endswith(("linear_attn.in_proj_a", "linear_attn.in_proj_b")):
        return "sparse_bf16"
    if module_name.endswith(("linear_attn.in_proj_qkv", "mlp.gate_proj", "mlp.up_proj", "self_attn.q_proj")):
        return "sparse_nvfp4"
    if module_name.endswith("mlp.down_proj"):
        return "sparse_bf16"
    if module_name.endswith((
        "linear_attn.in_proj_z",
        "linear_attn.out_proj",
        "self_attn.o_proj",
    )):
        return "sparse_bf16"
    if module_name.endswith(("self_attn.k_proj", "self_attn.v_proj")):
        return "marlin_nvfp4"
    return "bf16"


def _swh_decode_backend(module_name: str) -> str:
    """Data-driven optimal decode backend for Qwen3.5-9B at small M (<= 64).

    At small batch, marlin_nvfp4 (W4A16) dominates all but the tiniest N=32 layers.
    """
    if module_name.endswith(("linear_attn.in_proj_a", "linear_attn.in_proj_b")):
        return "bf16"
    return "marlin_nvfp4"


def _manual_prefill_backend(module_name: str, hybrid_scheme: str) -> str:
    is_m16384 = hybrid_scheme in ("manual_hybrid_m4", "manual_hybrid_m8", "manual_hybrid_m16")
    if module_name.endswith(("linear_attn.in_proj_a", "linear_attn.in_proj_b")):
        return "bf16"
    if module_name.endswith(("linear_attn.in_proj_qkv", "mlp.gate_proj", "mlp.up_proj")):
        return "dense_nvfp4" if is_m16384 else "sparse_nvfp4"
    if module_name.endswith("mlp.down_proj"):
        return "sparse_nvfp4"
    if module_name.endswith((
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "linear_attn.in_proj_z",
        "linear_attn.out_proj",
    )):
        return "sparse_nvfp4" if is_m16384 else "sparse_bf16"
    return "bf16"


def _manual_decode_backend(module_name: str, hybrid_scheme: str) -> str:
    if module_name.endswith(("self_attn.k_proj", "self_attn.v_proj", "linear_attn.in_proj_a", "linear_attn.in_proj_b")):
        return "bf16"
    if hybrid_scheme == "manual_hybrid_m8":
        if module_name.endswith(("linear_attn.in_proj_qkv", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")):
            return "bf16"
    if hybrid_scheme == "manual_hybrid_m4" and module_name.endswith("mlp.down_proj"):
        return "bf16"
    return "marlin_nvfp4"


def _load_wrapper():
    import importlib

    for module_name in (
        "fake.kernels.cutlass.cutlass_wrapper.cutlass_wrapper",
        "cutlass_wrapper",
    ):
        try:
            return importlib.import_module(module_name)
        except Exception:
            pass
    raise RuntimeError("CUTLASS wrapper package is not importable")


def _dtype_from_string(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    return getattr(torch, str(value).replace("torch.", ""), torch.bfloat16)
