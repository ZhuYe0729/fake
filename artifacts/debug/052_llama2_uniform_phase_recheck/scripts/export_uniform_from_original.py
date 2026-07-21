#!/usr/bin/env python3
"""Export the uniform NVFP4 format directly from original HF weights.

This is a forensic control: it intentionally bypasses the calibrated
``prepared/dense_nvfp4`` state used by the historical uniform baseline.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EXPORTER = ROOT / "artifacts/exports/vllm/baselines/llama2-7b-chat/scripts/export_uniform_vllm.py"
DEFAULT_MODEL = Path("/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf")
DEFAULT_CUTLASS = ROOT / "fake/kernels/cutlass/cutlass_wrapper"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cutlass-wrapper-path", type=Path, default=DEFAULT_CUTLASS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.cutlass_wrapper_path))
    spec = importlib.util.spec_from_file_location("uniform_export", EXPORTER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {EXPORTER}")
    exporter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = exporter
    spec.loader.exec_module(exporter)

    model_path = args.model_path.resolve()
    config = __import__("json").loads((model_path / "config.json").read_text())
    weight_map = exporter.read_weight_map(model_path)
    reader = exporter.LazyTensorReader(model_path, weight_map)

    class OriginalState:
        def __getitem__(self, name: str):
            return reader.get(name)

    exporter.load_prepared_state = lambda _root, _method: OriginalState()
    exporter.export_one_method(
        spec=exporter.METHOD_SPECS["dense_nvfp4"],
        model_path=model_path,
        prepared_root=Path("/unused"),
        output_root=args.output_root.resolve(),
        reader=reader,
        weight_map=weight_map,
        num_layers=int(config["num_hidden_layers"]),
        force=True,
    )


if __name__ == "__main__":
    main()
