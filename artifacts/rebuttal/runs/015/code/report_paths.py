"""Record names and local BF16 anchors for primary and supplemental campaigns."""
import json
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]


def record_id(policy):
    design = json.loads((RUN / 'configs/design.json').read_text())
    return design.get('speed_record_ids', {}).get(policy, policy)


def anchor_id(policy):
    design = json.loads((RUN / 'configs/design.json').read_text())
    return design.get('speed_anchor_ids', {}).get(policy, 'uniform_dense_bf16')


def speed_path(policy, number):
    return RUN / f'results/speed/node0/round{number}/{record_id(policy)}.json'
