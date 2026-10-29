# SPDX-License-Identifier: Apache-2.0
"""Artifact-local phase-switch Linear quantization method for vLLM tests."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
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
from vllm.model_executor.parameter import ModelWeightParameter

_DEFAULT_CUTLASS_WRAPPER_PATH = (
    "/home/agent/wja/project/my/cospaq/fake/fake/kernels/cutlass/"
    "cutlass_wrapper")

_ENABLED = False
_PHASE = "prefill"
_LAYERS: list[torch.nn.Module] = []
_SCHEDULED: set[int] = set()
_TRACE: list[dict[str, Any]] = []
_COPY_STREAM: Optional[torch.cuda.Stream] = None


def _now() -> float:
    return time.time()


def _trace(event: str, layer: torch.nn.Module, **kwargs: Any) -> None:
    if os.environ.get("PHASE_SWITCH_TRACE", "0") != "1":
        return
    _TRACE.append({
        "time": _now(),
        "event": event,
        "prefix": getattr(layer, "prefix", ""),
        **kwargs,
    })


def _ensure_cutlass_wrapper_path() -> None:
    path = os.environ.get(
        "PHASE_SWITCH_MYTEST_CUTLASS_WRAPPER_PATH",
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


def _load_unsharded_gpu(param: Parameter, loaded_weight: torch.Tensor,
                        *args: Any) -> None:
    del args
    if isinstance(param, UninitializedParameter):
        param.materialize(tuple(loaded_weight.shape),
                          device=_target_device(loaded_weight),
                          dtype=loaded_weight.dtype)
    if param.data.shape != loaded_weight.shape:
        raise ValueError("phase_switch_mytest parameter shape mismatch: "
                         f"param={tuple(param.data.shape)}, "
                         f"loaded={tuple(loaded_weight.shape)}")
    param.data.copy_(loaded_weight.to(device=param.data.device),
                     non_blocking=True)


def _load_decode_cpu(param: Parameter, loaded_weight: torch.Tensor,
                     *args: Any) -> None:
    del args
    layer = getattr(param, "phase_switch_layer")
    name = getattr(param, "phase_switch_name")
    if not hasattr(layer, "ps_decode_cpu"):
        layer.ps_decode_cpu = {}
    layer.ps_decode_cpu[name] = _pin_cpu(loaded_weight)


def _get_copy_stream() -> torch.cuda.Stream:
    global _COPY_STREAM
    if _COPY_STREAM is None:
        _COPY_STREAM = torch.cuda.Stream()
    return _COPY_STREAM


def _sync_copy_enabled() -> bool:
    return os.environ.get("PHASE_SWITCH_SYNC_COPY", "0") == "1"


def _gc_every_apply_enabled() -> bool:
    return os.environ.get("PHASE_SWITCH_GC_EVERY_APPLY", "1") == "1"


def _gc_decode_enabled() -> bool:
    return os.environ.get("PHASE_SWITCH_GC_DECODE", "1") == "1"


def _wait_once_enabled() -> bool:
    return os.environ.get("PHASE_SWITCH_WAIT_ONCE", "0") == "1"


def _release_prefill_enabled() -> bool:
    return os.environ.get("PHASE_SWITCH_RELEASE_PREFILL", "1") == "1"


def enable_phase_switch() -> None:
    global _ENABLED, _PHASE
    _ENABLED = True
    _PHASE = "prefill"
    _SCHEDULED.clear()
    _TRACE.clear()
    for layer in _LAYERS:
        layer.ps_decode_scheduled = False
        layer.ps_prefill_released = False
        layer.ps_decode_waited = False
    _TRACE.append({
        "time": _now(),
        "event": "enable",
        "num_layers": len(_LAYERS),
        "prefill_method": "dense_nvfp4",
        "decode_method": "w4a16_ours",
    })


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
        _TRACE.append({
            "time": _now(),
            "event": "enter_decode",
            "num_layers": len(_LAYERS),
        })


def _release_prefill_if_ready(layer: torch.nn.Module) -> None:
    if not _release_prefill_enabled():
        return
    if getattr(layer, "ps_prefill_released", False):
        return
    event = getattr(layer, "ps_decode_event", None)
    if event is None or not event.query():
        return
    device = layer.ps_prefill_weight.device
    layer.ps_prefill_weight = Parameter(torch.empty(0,
                                                    dtype=torch.uint8,
                                                    device=device),
                                        requires_grad=False)
    layer.ps_prefill_weight_scale = Parameter(torch.empty(0, device=device),
                                             requires_grad=False)
    layer.ps_prefill_weight_global_scale = Parameter(torch.empty(
        0, dtype=torch.float32, device=device),
                                                    requires_grad=False)
    layer.ps_prefill_released = True
    _trace("release_prefill", layer)


def _garbage_collect_ready() -> None:
    for layer in _LAYERS:
        _release_prefill_if_ready(layer)


def _schedule_decode_copy(layer: torch.nn.Module) -> None:
    if not _ENABLED or getattr(layer, "ps_decode_scheduled", False):
        return
    if not torch.cuda.is_available():
        raise RuntimeError("phase_switch_mytest requires CUDA")
    missing = {"packed_weight", "weight_scale", "weight_global_scale"} - set(
        getattr(layer, "ps_decode_cpu", {}).keys())
    if missing:
        raise RuntimeError(f"{layer.prefix} missing decode CPU tensors: "
                           f"{sorted(missing)}")

    if _sync_copy_enabled():
        layer.ps_decode_scheduled = True
        _SCHEDULED.add(id(layer))
        _trace("defer_decode_copy", layer)
        if len(_SCHEDULED) == len(_LAYERS):
            _sync_copy_all_decode_weights()
            _maybe_enter_decode()
        return

    if hasattr(layer, "ps_decode_packed_weight_gpu"):
        layer.ps_decode_event = torch.cuda.Event()
        layer.ps_decode_event.record(torch.cuda.current_stream())
        layer.ps_decode_scheduled = True
        _SCHEDULED.add(id(layer))
        _trace("reuse_decode_weight", layer)
        _maybe_enter_decode()
        return

    current_stream = torch.cuda.current_stream()
    prefill_done = torch.cuda.Event()
    prefill_done.record(current_stream)
    copy_stream = _get_copy_stream()
    with torch.cuda.stream(copy_stream):
        copy_stream.wait_event(prefill_done)
        layer.ps_decode_packed_weight_gpu = layer.ps_decode_cpu[
            "packed_weight"].to(device=layer.ps_prefill_weight.device,
                                non_blocking=True)
        layer.ps_decode_weight_scale_gpu = layer.ps_decode_cpu[
            "weight_scale"].to(device=layer.ps_prefill_weight.device,
                               non_blocking=True)
        layer.ps_decode_weight_global_scale_gpu = layer.ps_decode_cpu[
            "weight_global_scale"].to(device=layer.ps_prefill_weight.device,
                                      non_blocking=True)
        from cutlass_wrapper.marlin_nvfp4 import _make_workspace
        layer.ps_decode_workspace = _make_workspace(
            layer.ps_decode_packed_weight_gpu.device)
        layer.ps_decode_c_tmp = torch.empty(
            1,
            dtype=torch.float32,
            device=layer.ps_decode_packed_weight_gpu.device)
        layer.ps_decode_event = torch.cuda.Event()
        layer.ps_decode_event.record(copy_stream)

    layer.ps_decode_scheduled = True
    _SCHEDULED.add(id(layer))
    _trace("schedule_decode_copy", layer)
    _maybe_enter_decode()


def _sync_copy_one_decode_weight(layer: torch.nn.Module) -> None:
    layer.ps_decode_packed_weight_gpu = layer.ps_decode_cpu[
        "packed_weight"].to(device=layer.ps_prefill_weight.device,
                            non_blocking=False)
    layer.ps_decode_weight_scale_gpu = layer.ps_decode_cpu[
        "weight_scale"].to(device=layer.ps_prefill_weight.device,
                           non_blocking=False)
    layer.ps_decode_weight_global_scale_gpu = layer.ps_decode_cpu[
        "weight_global_scale"].to(device=layer.ps_prefill_weight.device,
                                  non_blocking=False)
    from cutlass_wrapper.marlin_nvfp4 import _make_workspace
    layer.ps_decode_workspace = _make_workspace(
        layer.ps_decode_packed_weight_gpu.device)
    layer.ps_decode_c_tmp = torch.empty(
        1,
        dtype=torch.float32,
        device=layer.ps_decode_packed_weight_gpu.device)
    layer.ps_decode_event = torch.cuda.Event()
    layer.ps_decode_event.record(torch.cuda.current_stream())


def _sync_copy_all_decode_weights() -> None:
    torch.cuda.synchronize()
    start = _now()
    _TRACE.append({
        "time": start,
        "event": "sync_decode_copy_start",
        "num_layers": len(_LAYERS),
    })
    for layer in _LAYERS:
        _sync_copy_one_decode_weight(layer)
        torch.cuda.synchronize()
        _release_prefill_if_ready(layer)
    end = _now()
    _TRACE.append({
        "time": end,
        "event": "sync_decode_copy_end",
        "num_layers": len(_LAYERS),
        "duration_s": end - start,
    })


def _wait_decode_ready(layer: torch.nn.Module) -> None:
    if _wait_once_enabled() and getattr(layer, "ps_decode_waited", False):
        return
    event = getattr(layer, "ps_decode_event", None)
    if event is None:
        raise RuntimeError(f"{layer.prefix} decode weights were not scheduled")
    torch.cuda.current_stream().wait_event(event)
    layer.ps_decode_waited = True
    _release_prefill_if_ready(layer)


@register_quantization_config("phase_switch_mytest")
class PhaseSwitchMyTestConfig(QuantizationConfig):
    """dense_nvfp4 prefill -> Marlin NVFP4 decode for offline tests."""

    def __init__(
        self,
        modules_to_not_convert: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self.modules_to_not_convert = modules_to_not_convert or ["lm_head"]

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "phase_switch_mytest"

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
    def from_config(cls, config: dict[str, Any]) -> "PhaseSwitchMyTestConfig":
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], ["lm_head"])
        return cls(modules_to_not_convert=modules_to_not_convert)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        if not isinstance(layer, LinearBase):
            return None
        if is_layer_skipped(prefix, self.modules_to_not_convert):
            return UnquantizedLinearMethod()
        return PhaseSwitchMyTestLinearMethod(self)


class PhaseSwitchMyTestLinearMethod(LinearMethodBase):

    def __init__(self, quant_config: PhaseSwitchMyTestConfig) -> None:
        self.quant_config = quant_config

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
        del input_size, output_size, params_dtype
        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError("phase_switch_mytest currently supports TP=1.")
        output_size_per_partition = sum(output_partition_sizes)
        if input_size_per_partition % 128 != 0:
            raise ValueError("phase_switch_mytest requires K divisible by 128")
        if output_size_per_partition % 64 != 0:
            raise ValueError("phase_switch_mytest requires N divisible by 64")

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.logical_widths = output_partition_sizes
        layer.ps_decode_cpu = {}
        layer.ps_decode_scheduled = False
        layer.ps_prefill_released = False
        _LAYERS.append(layer)

        weight_loader = extra_weight_attrs.get("weight_loader")
        prefill_weight = ModelWeightParameter(
            data=torch.empty(output_size_per_partition,
                             input_size_per_partition // 2,
                             dtype=torch.uint8),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("ps_prefill_weight", prefill_weight)

        prefill_scale = UninitializedParameter(requires_grad=False)
        prefill_scale.weight_loader = _load_unsharded_gpu
        layer.register_parameter("ps_prefill_weight_scale", prefill_scale)

        prefill_global = Parameter(torch.empty(1, dtype=torch.float32),
                                   requires_grad=False)
        prefill_global.weight_loader = weight_loader
        layer.register_parameter("ps_prefill_weight_global_scale",
                                 prefill_global)

        for param_name, decode_name in (
            ("ps_decode_packed_weight", "packed_weight"),
            ("ps_decode_weight_scale", "weight_scale"),
            ("ps_decode_weight_global_scale", "weight_global_scale"),
        ):
            param = Parameter(torch.empty(0), requires_grad=False)
            param.weight_loader = _load_decode_cpu
            param.phase_switch_layer = layer
            param.phase_switch_name = decode_name
            layer.register_parameter(param_name, param)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _ensure_cutlass_wrapper_path()
        from cutlass_wrapper.marlin_nvfp4 import _load_ext
        _load_ext()
        layer.ps_prefill_weight = Parameter(
            layer.ps_prefill_weight.data.contiguous().view(-1),
            requires_grad=False)
        layer.ps_prefill_weight_scale = Parameter(
            layer.ps_prefill_weight_scale.data.contiguous(),
            requires_grad=False)
        layer.ps_prefill_weight_global_scale = Parameter(
            layer.ps_prefill_weight_global_scale.max().to(
                torch.float32).reshape(1),
            requires_grad=False)

    def _apply_prefill(self, layer: torch.nn.Module, x: torch.Tensor,
                       bias: Optional[torch.Tensor]) -> torch.Tensor:
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
            layer.ps_prefill_weight_global_scale.to(torch.float32))
        out = nvfp4_gemm_bf16(a_packed, a_scale, layer.ps_prefill_weight,
                              layer.ps_prefill_weight_scale, alpha, m, n, k)
        if bias is not None:
            out = out + bias
        return out.reshape(*batch_shape, n)

    def _apply_decode(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        from cutlass_wrapper.marlin_nvfp4 import _load_ext
        _load_ext()
        _wait_decode_ready(layer)
        out = torch.ops.marlin_nvfp4_ext.marlin_linear_forward(
            x,
            layer.ps_decode_packed_weight_gpu,
            layer.ps_decode_weight_scale_gpu,
            layer.ps_decode_weight_global_scale_gpu,
            layer.ps_decode_workspace,
            layer.ps_decode_c_tmp,
            int(layer.output_size_per_partition),
            int(layer.input_size_per_partition),
        )
        if bias is not None:
            out = out + bias
        return out

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _ensure_cutlass_wrapper_path()
        if _gc_every_apply_enabled() and (_PHASE != "decode"
                                          or _gc_decode_enabled()):
            _garbage_collect_ready()
        if _ENABLED and _PHASE == "decode":
            _trace("apply_decode", layer)
            return self._apply_decode(layer, x, bias)

        _trace("apply_prefill", layer)
        out = self._apply_prefill(layer, x, bias)
        if _ENABLED:
            _schedule_decode_copy(layer)
        return out
