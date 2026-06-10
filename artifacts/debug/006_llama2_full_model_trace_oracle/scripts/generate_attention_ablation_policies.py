#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fake.kernels.offline_hybrid_policy import save_policy_json, write_policy_csv
from scripts.run_main_hybrid_policy_retest import SCENARIOS, ScenarioSpec, enumerate_linear_groups, make_decision, make_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    groups = enumerate_linear_groups("llama2-7b")
    scenario = ScenarioSpec(**SCENARIOS["normal_02"])
    toggles = ("self_attn.k_proj", "self_attn.q_proj", "self_attn.v_proj")
    for bits in itertools.product(("dense_bf16", "marlin_nvfp4"), repeat=len(toggles)):
        mapping = dict(zip(toggles, bits))
        name = "attn_" + "_".join(f"{key.split('.')[-1]}-{value.replace('_nvfp4','').replace('dense_','')}" for key, value in mapping.items())
        decisions = []
        for group in groups:
            if group.name.startswith("mlp."):
                prefill, decode = "dense_nvfp4", "marlin_nvfp4"
            elif group.name == "self_attn.o_proj":
                prefill, decode = "marlin_nvfp4", "marlin_nvfp4"
            elif group.name in mapping:
                prefill = decode = mapping[group.name]
            else:
                prefill = decode = "marlin_nvfp4"
            decisions.append(
                make_decision(
                    group,
                    selected_prefill=prefill,
                    selected_decode=decode,
                    total_ms=None,
                    prefill_ms=None,
                    decode_ms=None,
                    conversion_ms=0.0,
                    candidates=[],
                )
            )
        policy = make_policy(scenario, decisions)
        out = args.out_dir / name
        out.mkdir(parents=True, exist_ok=True)
        save_policy_json(policy, out / "policy.json")
        write_policy_csv(policy, out / "policy.csv")


if __name__ == "__main__":
    main()
