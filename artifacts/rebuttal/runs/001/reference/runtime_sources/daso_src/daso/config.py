from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    name: str
    path: Path
    layers: int
    hidden_size: int
    intermediate_size: int


@dataclass(frozen=True)
class WorkloadConfig:
    name: str
    batch_size: int
    input_tokens: int
    output_tokens: int
    gpu_memory_utilization: float = 0.8

    @property
    def scenario(self) -> str:
        return "prefill_only" if self.output_tokens <= 1 else "prefill_decode"

    @property
    def decode_steps(self) -> int:
        return max(0, self.output_tokens - 1)


@dataclass(frozen=True)
class BackendConfig:
    vllm_root: Path
    cutlass_wrapper_root: Path
    run_root: Path


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    workload: WorkloadConfig
    backend: BackendConfig
    raw: dict[str, Any]


def _path(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser().resolve()


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path).resolve()
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    model = raw["model"]
    workload = raw["workload"]
    backend = raw["backend"]
    return ExperimentConfig(
        model=ModelConfig(
            name=str(model["name"]), path=_path(str(model["path"])),
            layers=int(model["layers"]), hidden_size=int(model["hidden_size"]),
            intermediate_size=int(model["intermediate_size"]),
        ),
        workload=WorkloadConfig(
            name=str(workload["name"]), batch_size=int(workload["batch_size"]),
            input_tokens=int(workload["input_tokens"]),
            output_tokens=int(workload["output_tokens"]),
            gpu_memory_utilization=float(workload.get("gpu_memory_utilization", 0.8)),
        ),
        backend=BackendConfig(
            vllm_root=_path(str(backend["vllm_root"])),
            cutlass_wrapper_root=_path(str(backend["cutlass_wrapper_root"])),
            run_root=_path(str(backend["run_root"])),
        ),
        raw=raw,
    )

