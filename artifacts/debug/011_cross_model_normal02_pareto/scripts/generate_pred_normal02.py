#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEBUG_ROOT = Path(__file__).resolve().parents[1]
FAKE_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE_ROOT = FAKE_ROOT.parent
for path in (WORKSPACE_ROOT, FAKE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
SCRIPTS_ROOT = FAKE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from fake.kernels.offline_hybrid_policy import ScenarioSpec  # noqa: E402
from run_main_hybrid_policy_retest import (  # type: ignore  # noqa: E402
    MODELS,
    SCENARIOS,
    enumerate_linear_groups,
    pred_policy,
    write_policy_outputs,
)
from modeling.kernel_predictor import DEFAULT_MODEL_ROOT, KernelLatencyPredictor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate normal_02 predictor candidates without E2E GPU runs.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT / "latency_pred")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["llama31-8b", "qwen35-9b"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_name = "normal_02"
    scenario = ScenarioSpec(**SCENARIOS[scenario_name])
    predictor = KernelLatencyPredictor(model_root=DEFAULT_MODEL_ROOT)
    for model_key in args.models:
        groups = enumerate_linear_groups(model_key)
        out_dir = args.output_root / "pred" / scenario_name
        policy = pred_policy(model_key, scenario_name, groups, scenario, predictor, out_dir)
        write_policy_outputs(out_dir, model_key, scenario_name, policy)
        print(f"wrote {model_key}: groups={len(groups)} to {out_dir}")


if __name__ == "__main__":
    main()
