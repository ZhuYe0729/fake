# SPDX-License-Identifier: Apache-2.0
"""Artifact-local phase-heterogeneous Linear quantization for vLLM tests."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter, UninitializedParameter

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import (
    QuantizationMethods, register_quantization_config)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped)

_DEFAULT_CUTLASS_WRAPPER_PATH = (
    "/home/agent/wja/project/my/cospaq/fake/fake/kernels/cutlass/"
    "cutlass_wrapper")

SUPPORTED_METHODS = {
    "dense_bf16",
    "dense_nvfp4",
    "sparse_bf16",
    "sparse_nvfp4",
    "w4a16_ours",
}

_ENABLED = False
_PHASE = "prefill"
_LAYERS: list[torch.nn.Module] = []
_SCHEDULED: set[int] = set()
_TRACE: list[dict[str, Any]] = []
_COPY_STREAM: Optional[torch.cuda.Stream] = None
_TRACE_ENABLED = os.environ.get("PHASE_HETERO_TRACE", "0") == "1"
_SYNC_COPY_ENABLED = os.environ.get("PHASE_HETERO_SYNC_COPY", "0") == "1"
_RELEASE_PREFILL_ENABLED = (
    os.environ.get("PHASE_HETERO_RELEASE_PREFILL", "1") == "1")
_GC_EVERY_APPLY_ENABLED = (
    os.environ.get("PHASE_HETERO_GC_EVERY_APPLY", "1") == "1")
_GC_DECODE_ENABLED = os.environ.get("PHASE_HETERO_GC_DECODE", "0") == "1"
_WAIT_ONCE_ENABLED = os.environ.get("PHASE_HETERO_WAIT_ONCE", "1") == "1"


def _now() -> float:
    return time.time()


def _trace(event: str, layer: torch.nn.Module | None = None,
           **kwargs: Any) -> None:
    if not _TRACE_ENABLED:
        return
    row: dict[str, Any] = {"time": _now(), "event": event, **kwargs}
    if layer is not None:
        row["prefix"] = getattr(layer, "prefix", "")
        row["prefill_method"] = getattr(layer, "ph_prefill_method", "")
        row["decode_method"] = getattr(layer, "ph_decode_method", "")
    _TRACE.append(row)


def _ensure_cutlass_wrapper_path() -> None:
    path = os.environ.get(
        "PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH",
        os.environ.get("NVFP4_MYTEST_CUTLASS_WRAPPER_PATH",
                       _DEFAULT_CUTLASS_WRAPPER_PATH))
    if path and path not in sys.path:
        sys.path.insert(0, path)


def _target_device(tensor: torch.Tensor) -> torch.device:
    if tensor.device.type == "cpu" and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return tensor.device


def _pin_cpu(tensor: torch.Tensor) -> torch.Tensor:
    cpu = tensor.detach().cpu().contiguous()
    if torch.cuda.is_available():
        return cpu.pin_memory()
    return cpu


def _load_prefill_gpu(param: Parameter, loaded_weight: torch.Tensor,
                      *args: Any) -> None:
    del args
    if isinstance(param, UninitializedParameter):
        param.materialize(tuple(loaded_weight.shape),
                          device=_target_device(loaded_weight),
                          dtype=loaded_weight.dtype)
    if param.data.shape != loaded_weight.shape:
        raise ValueError("phase_hetero_mytest prefill shape mismatch: "
                         f"param={tuple(param.data.shape)}, "
                         f"loaded={tuple(loaded_weight.shape)}")
    param.data.copy_(loaded_weight.to(device=param.data.device),
                     non_blocking=True)


def _load_decode_cpu(param: Parameter, loaded_weight: torch.Tensor,
                     *args: Any) -> None:
    del args
    layer = getattr(param, "phase_hetero_layer")
    name = getattr(param, "phase_hetero_name")
    layer.ph_decode_cpu[name] = _pin_cpu(loaded_weight)


def _get_copy_stream() -> torch.cuda.Stream:
    global _COPY_STREAM
    if _COPY_STREAM is None:
        _COPY_STREAM = torch.cuda.Stream()
    return _COPY_STREAM


def _sync_copy_enabled() -> bool:
    return _SYNC_COPY_ENABLED


def _release_prefill_enabled() -> bool:
    return _RELEASE_PREFILL_ENABLED


def _gc_every_apply_enabled() -> bool:
    return _GC_EVERY_APPLY_ENABLED


def _gc_decode_enabled() -> bool:
    return _GC_DECODE_ENABLED


def _wait_once_enabled() -> bool:
    return _WAIT_ONCE_ENABLED


def _get_tp_size() -> int:
    try:
        return get_tensor_model_parallel_world_size()
    except AssertionError:
        return 1


@dataclass(frozen=True)
class PhaseMethodPolicy:
    prefill_method: str
    decode_method: str


class LinearAdapter:
    method: str
    names: tuple[str, ...]

    def validate_shape(self, layer: torch.nn.Module) -> None:
        raise NotImplementedError

    def finalize_prefill(self, layer: torch.nn.Module) -> None:
        raise NotImplementedError

    def finalize_decode(self, layer: torch.nn.Module) -> None:
        raise NotImplementedError

    def release_prefill(self, layer: torch.nn.Module) -> None:
        for name in self.names:
            setattr(layer, f"ph_prefill_{name}",
                    Parameter(torch.empty(0, device="cuda"),
                              requires_grad=False))

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: Optional[torch.Tensor], decode: bool) -> torch.Tensor:
        raise NotImplementedError


class DenseBF16Adapter(LinearAdapter):
    method = "dense_bf16"
    names = ("weight", )

    def validate_shape(self, layer: torch.nn.Module) -> None:
        return

    def finalize_prefill(self, layer: torch.nn.Module) -> None:
        layer.ph_prefill_weight = Parameter(
            layer.ph_prefill_weight.data.to(torch.bfloat16).contiguous(),
            requires_grad=False)

    def finalize_decode(self, layer: torch.nn.Module) -> None:
        layer.ph_decode_weight_gpu = layer.ph_decode_weight_gpu.to(
            torch.bfloat16).contiguous()

    def release_prefill(self, layer: torch.nn.Module) -> None:
        device = layer.ph_prefill_weight.device
        layer.ph_prefill_weight = Parameter(torch.empty(0,
                                                       dtype=torch.bfloat16,
                                                       device=device),
                                           requires_grad=False)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: Optional[torch.Tensor], decode: bool) -> torch.Tensor:
        weight = (layer.ph_decode_weight_gpu
                  if decode else layer.ph_prefill_weight)
        batch_shape = x.shape[:-1]
        x_flat = x.to(dtype=torch.bfloat16).reshape(
            -1, layer.input_size_per_partition)
        out = F.linear(x_flat, weight, bias)
        return out.reshape(*batch_shape, layer.output_size_per_partition)


class DenseNVFP4Adapter(LinearAdapter):
    method = "dense_nvfp4"
    names = ("weight", "weight_scale", "weight_global_scale")

    def validate_shape(self, layer: torch.nn.Module) -> None:
        if layer.input_size_per_partition % 32 != 0:
            raise ValueError("dense_nvfp4 requires K divisible by 32")
        if layer.output_size_per_partition % 32 != 0:
            raise ValueError("dense_nvfp4 requires N divisible by 32")

    def finalize_prefill(self, layer: torch.nn.Module) -> None:
        layer.ph_prefill_weight = Parameter(
            layer.ph_prefill_weight.data.contiguous().view(-1),
            requires_grad=False)
        layer.ph_prefill_weight_scale = Parameter(
            layer.ph_prefill_weight_scale.data.contiguous(),
            requires_grad=False)
        layer.ph_prefill_weight_global_scale = Parameter(
            layer.ph_prefill_weight_global_scale.data.max().to(
                torch.float32).reshape(1),
            requires_grad=False)

    def finalize_decode(self, layer: torch.nn.Module) -> None:
        layer.ph_decode_weight_gpu = layer.ph_decode_weight_gpu.contiguous(
        ).view(-1)
        layer.ph_decode_weight_scale_gpu = (
            layer.ph_decode_weight_scale_gpu.contiguous())
        layer.ph_decode_weight_global_scale_gpu = (
            layer.ph_decode_weight_global_scale_gpu.max().to(
                torch.float32).reshape(1))

    def release_prefill(self, layer: torch.nn.Module) -> None:
        device = layer.ph_prefill_weight.device
        layer.ph_prefill_weight = Parameter(torch.empty(0,
                                                       dtype=torch.uint8,
                                                       device=device),
                                           requires_grad=False)
        layer.ph_prefill_weight_scale = Parameter(torch.empty(0,
                                                             device=device),
                                                 requires_grad=False)
        layer.ph_prefill_weight_global_scale = Parameter(
            torch.empty(0, dtype=torch.float32, device=device),
            requires_grad=False)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: Optional[torch.Tensor], decode: bool) -> torch.Tensor:
        from cutlass_wrapper import (can_use_cutlass_nvfp4, nvfp4_gemm_bf16,
                                     pack_nvfp4_a)

        if decode:
            weight = layer.ph_decode_weight_gpu
            weight_scale = layer.ph_decode_weight_scale_gpu
            weight_global_scale = layer.ph_decode_weight_global_scale_gpu
        else:
            weight = layer.ph_prefill_weight
            weight_scale = layer.ph_prefill_weight_scale
            weight_global_scale = layer.ph_prefill_weight_global_scale

        batch_shape = x.shape[:-1]
        x_flat = x.to(dtype=torch.bfloat16).reshape(
            -1, layer.input_size_per_partition).contiguous()
        m = int(x_flat.shape[0])
        n = int(layer.output_size_per_partition)
        k = int(layer.input_size_per_partition)
        if not can_use_cutlass_nvfp4(m, n, k):
            raise RuntimeError("dense NVFP4 cannot implement "
                               f"M={m}, N={n}, K={k}")
        a_packed, a_scale, a_global_scale = pack_nvfp4_a(x_flat)
        alpha = torch.reciprocal(a_global_scale *
                                 weight_global_scale.to(torch.float32))
        out = nvfp4_gemm_bf16(a_packed, a_scale, weight, weight_scale, alpha,
                              m, n, k)
        if bias is not None:
            out = out + bias
        return out.reshape(*batch_shape, n)


class SparseBF16Adapter(LinearAdapter):
    method = "sparse_bf16"
    names = ("sparse_weight", "metadata")

    def validate_shape(self, layer: torch.nn.Module) -> None:
        if layer.input_size_per_partition % 64 != 0:
            raise ValueError("sparse_bf16 requires K divisible by 64")
        if layer.output_size_per_partition % 8 != 0:
            raise ValueError("sparse_bf16 requires N divisible by 8")

    def finalize_prefill(self, layer: torch.nn.Module) -> None:
        layer.ph_prefill_sparse_weight = Parameter(
            layer.ph_prefill_sparse_weight.data.contiguous(),
            requires_grad=False)
        layer.ph_prefill_metadata = Parameter(
            layer.ph_prefill_metadata.data.contiguous(), requires_grad=False)

    def finalize_decode(self, layer: torch.nn.Module) -> None:
        layer.ph_decode_sparse_weight_gpu = (
            layer.ph_decode_sparse_weight_gpu.contiguous())
        layer.ph_decode_metadata_gpu = layer.ph_decode_metadata_gpu.contiguous()

    def release_prefill(self, layer: torch.nn.Module) -> None:
        device = layer.ph_prefill_sparse_weight.device
        layer.ph_prefill_sparse_weight = Parameter(torch.empty(0,
                                                              device=device),
                                                  requires_grad=False)
        layer.ph_prefill_metadata = Parameter(torch.empty(0, device=device),
                                             requires_grad=False)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: Optional[torch.Tensor], decode: bool) -> torch.Tensor:
        from cutlass_wrapper import (can_use_cusparselt_sparse_bf16,
                                     sparse_bf16_gemm_bf16)

        if decode:
            sparse_weight = layer.ph_decode_sparse_weight_gpu
            metadata = layer.ph_decode_metadata_gpu
        else:
            sparse_weight = layer.ph_prefill_sparse_weight
            metadata = layer.ph_prefill_metadata

        batch_shape = x.shape[:-1]
        x_flat = x.to(dtype=torch.bfloat16).reshape(
            -1, layer.input_size_per_partition).contiguous()
        tokens = int(x_flat.shape[0])
        padded_tokens = ((tokens + 7) // 8) * 8
        if padded_tokens != tokens:
            x_flat = F.pad(x_flat, (0, 0, 0, padded_tokens - tokens))

        m = int(layer.output_size_per_partition)
        n = int(padded_tokens)
        k = int(layer.input_size_per_partition)
        if not can_use_cusparselt_sparse_bf16(m, n, k):
            raise RuntimeError("sparse BF16 cannot implement "
                               f"M={m}, N={n}, K={k}")
        out = sparse_bf16_gemm_bf16(m, sparse_weight, metadata, x_flat, 1.0,
                                    backend="cusparselt")
        if tokens != n:
            out = out[:tokens]
        if bias is not None:
            out = out + bias
        return out.reshape(*batch_shape, m)


class SparseNVFP4Adapter(LinearAdapter):
    method = "sparse_nvfp4"
    names = ("sparse_weight", "metadata", "weight_scale",
             "weight_global_scale")

    def validate_shape(self, layer: torch.nn.Module) -> None:
        if layer.input_size_per_partition % 64 != 0:
            raise ValueError("sparse_nvfp4 requires K divisible by 64")

    def finalize_prefill(self, layer: torch.nn.Module) -> None:
        layer.ph_prefill_sparse_weight = Parameter(
            layer.ph_prefill_sparse_weight.data.contiguous(),
            requires_grad=False)
        layer.ph_prefill_metadata = Parameter(
            layer.ph_prefill_metadata.data.contiguous(), requires_grad=False)
        layer.ph_prefill_weight_scale = Parameter(
            layer.ph_prefill_weight_scale.data.contiguous(),
            requires_grad=False)
        layer.ph_prefill_weight_global_scale = Parameter(
            layer.ph_prefill_weight_global_scale.data.to(
                torch.float32).reshape(1),
            requires_grad=False)

    def finalize_decode(self, layer: torch.nn.Module) -> None:
        layer.ph_decode_sparse_weight_gpu = (
            layer.ph_decode_sparse_weight_gpu.contiguous())
        layer.ph_decode_metadata_gpu = layer.ph_decode_metadata_gpu.contiguous()
        layer.ph_decode_weight_scale_gpu = (
            layer.ph_decode_weight_scale_gpu.contiguous())
        layer.ph_decode_weight_global_scale_gpu = (
            layer.ph_decode_weight_global_scale_gpu.to(torch.float32).reshape(1))

    def release_prefill(self, layer: torch.nn.Module) -> None:
        device = layer.ph_prefill_sparse_weight.device
        for name in self.names:
            setattr(layer, f"ph_prefill_{name}",
                    Parameter(torch.empty(0, device=device),
                              requires_grad=False))

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: Optional[torch.Tensor], decode: bool) -> torch.Tensor:
        from cutlass_wrapper import (can_use_cutlass_sparse_nvfp4,
                                     pack_sparse_nvfp4_b,
                                     sparse_nvfp4_gemm_bf16)

        if decode:
            sparse_weight = layer.ph_decode_sparse_weight_gpu
            metadata = layer.ph_decode_metadata_gpu
            weight_scale = layer.ph_decode_weight_scale_gpu
            weight_global_scale = layer.ph_decode_weight_global_scale_gpu
        else:
            sparse_weight = layer.ph_prefill_sparse_weight
            metadata = layer.ph_prefill_metadata
            weight_scale = layer.ph_prefill_weight_scale
            weight_global_scale = layer.ph_prefill_weight_global_scale

        batch_shape = x.shape[:-1]
        x_flat = x.to(dtype=torch.bfloat16).reshape(
            -1, layer.input_size_per_partition).contiguous()
        tokens = int(x_flat.shape[0])
        padded_tokens = ((tokens + 31) // 32) * 32
        if padded_tokens != tokens:
            x_flat = F.pad(x_flat, (0, 0, 0, padded_tokens - tokens))

        m = int(layer.output_size_per_partition)
        n = int(padded_tokens)
        k = int(layer.input_size_per_partition)
        if not can_use_cutlass_sparse_nvfp4(m, n, k):
            raise RuntimeError("sparse NVFP4 cannot implement "
                               f"M={m}, N={n}, K={k}")
        b_packed, b_scale, b_global_scale = pack_sparse_nvfp4_b(x_flat)
        alpha = torch.reciprocal(weight_global_scale.to(torch.float32) *
                                 b_global_scale)
        out_storage = sparse_nvfp4_gemm_bf16(m, sparse_weight, metadata,
                                             b_packed, weight_scale, b_scale,
                                             alpha)
        out = out_storage.as_strided((n, m), (m, 1)).contiguous()
        if tokens != n:
            out = out[:tokens]
        if bias is not None:
            out = out + bias
        return out.reshape(*batch_shape, m)


class W4A16MarlinAdapter(LinearAdapter):
    method = "w4a16_ours"
    names = ("packed_weight", "weight_scale", "weight_global_scale")

    def validate_shape(self, layer: torch.nn.Module) -> None:
        if layer.input_size_per_partition % 128 != 0:
            raise ValueError("w4a16_ours requires K divisible by 128")
        if layer.output_size_per_partition % 64 != 0:
            raise ValueError("w4a16_ours requires N divisible by 64")

    def finalize_prefill(self, layer: torch.nn.Module) -> None:
        from cutlass_wrapper.marlin_nvfp4 import _load_ext, _make_workspace
        _load_ext()
        layer.ph_prefill_packed_weight = Parameter(
            layer.ph_prefill_packed_weight.data.contiguous(),
            requires_grad=False)
        layer.ph_prefill_weight_scale = Parameter(
            layer.ph_prefill_weight_scale.data.contiguous(),
            requires_grad=False)
        layer.ph_prefill_weight_global_scale = Parameter(
            layer.ph_prefill_weight_global_scale.data.contiguous(),
            requires_grad=False)
        layer.ph_prefill_workspace = _make_workspace(
            layer.ph_prefill_packed_weight.device)
        layer.ph_prefill_c_tmp = torch.empty(
            1, dtype=torch.float32, device=layer.ph_prefill_packed_weight.device)

    def finalize_decode(self, layer: torch.nn.Module) -> None:
        from cutlass_wrapper.marlin_nvfp4 import _load_ext, _make_workspace
        _load_ext()
        layer.ph_decode_packed_weight_gpu = (
            layer.ph_decode_packed_weight_gpu.contiguous())
        layer.ph_decode_weight_scale_gpu = (
            layer.ph_decode_weight_scale_gpu.contiguous())
        layer.ph_decode_weight_global_scale_gpu = (
            layer.ph_decode_weight_global_scale_gpu.contiguous())
        layer.ph_decode_workspace = _make_workspace(
            layer.ph_decode_packed_weight_gpu.device)
        layer.ph_decode_c_tmp = torch.empty(
            1, dtype=torch.float32,
            device=layer.ph_decode_packed_weight_gpu.device)

    def release_prefill(self, layer: torch.nn.Module) -> None:
        device = layer.ph_prefill_packed_weight.device
        for name in self.names:
            setattr(layer, f"ph_prefill_{name}",
                    Parameter(torch.empty(0, device=device),
                              requires_grad=False))
        layer.ph_prefill_workspace = None
        layer.ph_prefill_c_tmp = None

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: Optional[torch.Tensor], decode: bool) -> torch.Tensor:
        from cutlass_wrapper.marlin_nvfp4 import _load_ext
        _load_ext()
        if decode:
            packed_weight = layer.ph_decode_packed_weight_gpu
            weight_scale = layer.ph_decode_weight_scale_gpu
            weight_global_scale = layer.ph_decode_weight_global_scale_gpu
            workspace = layer.ph_decode_workspace
            c_tmp = layer.ph_decode_c_tmp
        else:
            packed_weight = layer.ph_prefill_packed_weight
            weight_scale = layer.ph_prefill_weight_scale
            weight_global_scale = layer.ph_prefill_weight_global_scale
            workspace = layer.ph_prefill_workspace
            c_tmp = layer.ph_prefill_c_tmp
        out = torch.ops.marlin_nvfp4_ext.marlin_linear_forward(
            x, packed_weight, weight_scale, weight_global_scale, workspace,
            c_tmp, int(layer.output_size_per_partition),
            int(layer.input_size_per_partition))
        if bias is not None:
            out = out + bias
        return out


ADAPTERS: dict[str, LinearAdapter] = {
    "dense_bf16": DenseBF16Adapter(),
    "dense_nvfp4": DenseNVFP4Adapter(),
    "sparse_bf16": SparseBF16Adapter(),
    "sparse_nvfp4": SparseNVFP4Adapter(),
    "w4a16_ours": W4A16MarlinAdapter(),
}


def _register_prefill_param(layer: torch.nn.Module, name: str) -> None:
    param = UninitializedParameter(requires_grad=False)
    param.weight_loader = _load_prefill_gpu
    layer.register_parameter(f"ph_prefill_{name}", param)


def _register_decode_param(layer: torch.nn.Module, name: str) -> None:
    param = Parameter(torch.empty(0), requires_grad=False)
    param.weight_loader = _load_decode_cpu
    param.phase_hetero_layer = layer
    param.phase_hetero_name = name
    layer.register_parameter(f"ph_decode_{name}", param)


def enable_phase_hetero() -> None:
    global _ENABLED, _PHASE
    _ENABLED = True
    _PHASE = "prefill"
    _SCHEDULED.clear()
    _TRACE.clear()
    for layer in _LAYERS:
        layer.ph_decode_scheduled = False
        layer.ph_prefill_released = False
        layer.ph_decode_waited = False
    _trace("enable",
           None,
           num_layers=len(_LAYERS),
           supported_methods=sorted(SUPPORTED_METHODS))


def enable_phase_switch() -> None:
    enable_phase_hetero()


def current_phase() -> str:
    return _PHASE


def dump_trace(path: str | os.PathLike[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump({
            "phase": _PHASE,
            "num_layers": len(_LAYERS),
            "num_scheduled": len(_SCHEDULED),
            "trace": _TRACE,
        }, f, indent=2)
        f.write("\n")


def _maybe_enter_decode() -> None:
    global _PHASE
    if _PHASE == "prefill" and len(_SCHEDULED) == len(_LAYERS):
        _PHASE = "decode"
        _trace("enter_decode", None, num_layers=len(_LAYERS))


def _release_prefill_if_ready(layer: torch.nn.Module) -> None:
    if not _release_prefill_enabled():
        return
    if layer.ph_prefill_method == layer.ph_decode_method:
        return
    if getattr(layer, "ph_prefill_released", False):
        return
    event = getattr(layer, "ph_decode_event", None)
    if event is None or not event.query():
        return
    layer.ph_prefill_adapter.release_prefill(layer)
    layer.ph_prefill_released = True
    _trace("release_prefill", layer)


def _garbage_collect_ready() -> None:
    for layer in _LAYERS:
        _release_prefill_if_ready(layer)


def _materialize_decode_from_cpu(layer: torch.nn.Module,
                                 non_blocking: bool) -> None:
    device = _decode_target_device(layer)
    for name in layer.ph_decode_adapter.names:
        tensor = layer.ph_decode_cpu[name].to(device=device,
                                              non_blocking=non_blocking)
        setattr(layer, f"ph_decode_{name}_gpu", tensor)
    layer.ph_decode_adapter.finalize_decode(layer)


def _decode_target_device(layer: torch.nn.Module) -> torch.device:
    for name in layer.ph_prefill_adapter.names:
        tensor = getattr(layer, f"ph_prefill_{name}", None)
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda":
            return tensor.device
    return torch.device("cuda", torch.cuda.current_device())


def _reuse_prefill_for_decode(layer: torch.nn.Module) -> None:
    for name in layer.ph_prefill_adapter.names:
        tensor = getattr(layer, f"ph_prefill_{name}")
        setattr(layer, f"ph_decode_{name}_gpu", tensor)
    if layer.ph_decode_method == "w4a16_ours":
        layer.ph_decode_workspace = layer.ph_prefill_workspace
        layer.ph_decode_c_tmp = layer.ph_prefill_c_tmp
    layer.ph_decode_event = torch.cuda.Event()
    layer.ph_decode_event.record(torch.cuda.current_stream())


def _schedule_decode_copy(layer: torch.nn.Module) -> None:
    if not _ENABLED or getattr(layer, "ph_decode_scheduled", False):
        return
    if not torch.cuda.is_available():
        raise RuntimeError("phase_hetero_mytest requires CUDA")

    layer.ph_decode_scheduled = True
    _SCHEDULED.add(id(layer))

    if layer.ph_prefill_method == layer.ph_decode_method:
        _reuse_prefill_for_decode(layer)
        _trace("reuse_decode_weight", layer)
        _maybe_enter_decode()
        return

    missing = set(layer.ph_decode_adapter.names) - set(layer.ph_decode_cpu)
    if missing:
        raise RuntimeError(f"{layer.prefix} missing decode CPU tensors: "
                           f"{sorted(missing)}")

    if _sync_copy_enabled():
        _trace("defer_decode_copy", layer)
        if len(_SCHEDULED) == len(_LAYERS):
            _sync_copy_all_decode_weights()
            _maybe_enter_decode()
        return

    current_stream = torch.cuda.current_stream()
    prefill_done = torch.cuda.Event()
    prefill_done.record(current_stream)
    copy_stream = _get_copy_stream()
    with torch.cuda.stream(copy_stream):
        copy_stream.wait_event(prefill_done)
        _materialize_decode_from_cpu(layer, non_blocking=True)
        layer.ph_decode_event = torch.cuda.Event()
        layer.ph_decode_event.record(copy_stream)

    _trace("schedule_decode_copy", layer)
    _maybe_enter_decode()


def _sync_copy_all_decode_weights() -> None:
    torch.cuda.synchronize()
    start = _now()
    _trace("sync_decode_copy_start", None, num_layers=len(_LAYERS))
    for layer in _LAYERS:
        if layer.ph_prefill_method != layer.ph_decode_method:
            _materialize_decode_from_cpu(layer, non_blocking=False)
            layer.ph_decode_event = torch.cuda.Event()
            layer.ph_decode_event.record(torch.cuda.current_stream())
        _release_prefill_if_ready(layer)
        torch.cuda.synchronize()
    _trace("sync_decode_copy_end",
           None,
           num_layers=len(_LAYERS),
           duration_s=_now() - start)


def _wait_decode_ready(layer: torch.nn.Module) -> None:
    event = getattr(layer, "ph_decode_event", None)
    if event is None:
        raise RuntimeError(f"{layer.prefix} decode weights were not scheduled")
    torch.cuda.current_stream().wait_event(event)
    layer.ph_decode_waited = True
    _release_prefill_if_ready(layer)


@register_quantization_config("phase_hetero_mytest")
class PhaseHeteroMyTestConfig(QuantizationConfig):
    """Per-Linear prefill/decode method switching for offline tests."""

    def __init__(
        self,
        method_map: Optional[dict[str, Any]] = None,
        default_prefill_method: str = "dense_nvfp4",
        default_decode_method: str = "w4a16_ours",
        modules_to_not_convert: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self.method_map = dict(method_map or {})
        self.default_prefill_method = default_prefill_method
        self.default_decode_method = default_decode_method
        self.modules_to_not_convert = modules_to_not_convert or ["lm_head"]
        self._validate_method("default_prefill_method", default_prefill_method)
        self._validate_method("default_decode_method", default_decode_method)
        for prefix, value in self.method_map.items():
            if not isinstance(prefix, str) or not prefix:
                raise ValueError("phase_hetero_mytest method_map keys must be "
                                 "non-empty vLLM Linear prefixes")
            self._policy_from_value(prefix, value)

    @staticmethod
    def _validate_method(field: str, method: str) -> None:
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"{field}={method!r} is unsupported; expected "
                             f"one of {sorted(SUPPORTED_METHODS)}")

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "phase_hetero_mytest"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 120

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PhaseHeteroMyTestConfig":
        method_map = cls.get_from_keys_or(config, ["method_map"], {})
        default_prefill_method = cls.get_from_keys_or(
            config, ["default_prefill_method", "prefill_method"],
            "dense_nvfp4")
        default_decode_method = cls.get_from_keys_or(
            config, ["default_decode_method", "decode_method"], "w4a16_ours")
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], ["lm_head"])
        if not isinstance(method_map, dict):
            raise ValueError("phase_hetero_mytest method_map must be a dict")
        return cls(method_map=method_map,
                   default_prefill_method=default_prefill_method,
                   default_decode_method=default_decode_method,
                   modules_to_not_convert=modules_to_not_convert)

    def _policy_from_value(self, prefix: str, value: Any) -> PhaseMethodPolicy:
        if isinstance(value, str):
            prefill_method = value
            decode_method = value
        elif isinstance(value, dict):
            prefill_method = str(
                value.get("prefill_method", self.default_prefill_method))
            decode_method = str(
                value.get("decode_method", self.default_decode_method))
        else:
            raise ValueError(f"method_map[{prefix!r}] must be a string or dict")
        self._validate_method(f"{prefix}.prefill_method", prefill_method)
        self._validate_method(f"{prefix}.decode_method", decode_method)
        return PhaseMethodPolicy(prefill_method=prefill_method,
                                 decode_method=decode_method)

    def get_policy_for_prefix(self, prefix: str) -> PhaseMethodPolicy:
        if prefix in self.method_map:
            return self._policy_from_value(prefix, self.method_map[prefix])
        return PhaseMethodPolicy(self.default_prefill_method,
                                 self.default_decode_method)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        if not isinstance(layer, LinearBase):
            return None
        if is_layer_skipped(prefix, self.modules_to_not_convert):
            return UnquantizedLinearMethod()
        policy = self.get_policy_for_prefix(prefix)
        return PhaseHeteroMyTestLinearMethod(self, policy)


class PhaseHeteroMyTestLinearMethod(LinearMethodBase):

    def __init__(self, quant_config: PhaseHeteroMyTestConfig,
                 policy: PhaseMethodPolicy) -> None:
        self.quant_config = quant_config
        self.policy = policy
        self.prefill_adapter = ADAPTERS[policy.prefill_method]
        self.decode_adapter = ADAPTERS[policy.decode_method]
        self.prefill_apply = (self._apply_dense_nvfp4_prefill
                              if policy.prefill_method == "dense_nvfp4" else
                              self._apply_prefill_adapter)
        self.decode_apply = (self._apply_w4a16_decode
                             if policy.decode_method == "w4a16_ours" else
                             self._apply_decode_adapter)

    def _apply_prefill_adapter(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self.prefill_adapter.apply(layer, x, bias, False)

    def _apply_decode_adapter(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self.decode_adapter.apply(layer, x, bias, True)

    @staticmethod
    def _apply_dense_nvfp4_prefill(
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        from cutlass_wrapper import (can_use_cutlass_nvfp4, nvfp4_gemm_bf16,
                                     pack_nvfp4_a)

        batch_shape = x.shape[:-1]
        x_flat = x.to(dtype=torch.bfloat16).reshape(
            -1, layer.input_size_per_partition).contiguous()
        m = int(x_flat.shape[0])
        n = int(layer.output_size_per_partition)
        k = int(layer.input_size_per_partition)
        if not can_use_cutlass_nvfp4(m, n, k):
            raise RuntimeError("dense NVFP4 cannot implement "
                               f"M={m}, N={n}, K={k}")
        a_packed, a_scale, a_global_scale = pack_nvfp4_a(x_flat)
        alpha = torch.reciprocal(
            a_global_scale *
            layer.ph_prefill_weight_global_scale.to(torch.float32))
        out = nvfp4_gemm_bf16(a_packed, a_scale, layer.ph_prefill_weight,
                              layer.ph_prefill_weight_scale, alpha, m, n, k)
        if bias is not None:
            out = out + bias
        return out.reshape(*batch_shape, n)

    @staticmethod
    def _apply_w4a16_decode(
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        from cutlass_wrapper.marlin_nvfp4 import _load_ext

        _load_ext()
        out = torch.ops.marlin_nvfp4_ext.marlin_linear_forward(
            x, layer.ph_decode_packed_weight_gpu,
            layer.ph_decode_weight_scale_gpu,
            layer.ph_decode_weight_global_scale_gpu,
            layer.ph_decode_workspace, layer.ph_decode_c_tmp,
            int(layer.output_size_per_partition),
            int(layer.input_size_per_partition))
        if bias is not None:
            out = out + bias
        return out

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        del input_size, output_size, params_dtype, extra_weight_attrs
        if _get_tp_size() != 1:
            raise ValueError("phase_hetero_mytest currently supports TP=1")

        output_size_per_partition = sum(output_partition_sizes)
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.logical_widths = output_partition_sizes
        layer.ph_prefill_method = self.policy.prefill_method
        layer.ph_decode_method = self.policy.decode_method
        layer.ph_prefill_adapter = self.prefill_adapter
        layer.ph_decode_adapter = self.decode_adapter
        layer.ph_decode_cpu = {}
        layer.ph_decode_scheduled = False
        layer.ph_prefill_released = False
        layer.ph_decode_waited = False
        _LAYERS.append(layer)

        layer.ph_prefill_adapter.validate_shape(layer)
        layer.ph_decode_adapter.validate_shape(layer)
        for name in layer.ph_prefill_adapter.names:
            _register_prefill_param(layer, name)
        if layer.ph_prefill_method != layer.ph_decode_method:
            for name in layer.ph_decode_adapter.names:
                _register_decode_param(layer, name)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _ensure_cutlass_wrapper_path()
        layer.ph_prefill_adapter.finalize_prefill(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _ensure_cutlass_wrapper_path()
        if _ENABLED and _PHASE == "decode":
            if _GC_EVERY_APPLY_ENABLED and _GC_DECODE_ENABLED:
                _garbage_collect_ready()
            if not (_WAIT_ONCE_ENABLED and layer.ph_decode_waited):
                _wait_decode_ready(layer)
            if _TRACE_ENABLED:
                _trace("apply_decode", layer)
            return self.decode_apply(layer, x, bias)

        if _GC_EVERY_APPLY_ENABLED:
            _garbage_collect_ready()
        if _TRACE_ENABLED:
            _trace("apply_prefill", layer)
        out = self.prefill_apply(layer, x, bias)
        if _ENABLED:
            _schedule_decode_copy(layer)
        return out
