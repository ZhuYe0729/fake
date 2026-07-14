#!/usr/bin/env python3
"""Project a quality-calibration policy onto phase-runtime legal method pairs."""
from __future__ import annotations
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGAL = {(x, x) for x in ('dense_bf16', 'dense_nvfp4', 'sparse_bf16', 'sparse_nvfp4', 'w4a16_ours')}
LEGAL |= {('dense_nvfp4', 'w4a16_ours'), ('w4a16_ours', 'dense_nvfp4')}

def main():
    p = argparse.ArgumentParser(); p.add_argument('policy_id'); a = p.parse_args()
    src = ROOT / 'policies/prefill_decode' / f'{a.policy_id}.json'
    policy = json.loads(src.read_text()); repaired = 0
    for item in policy['method_map'].values():
        # Sparse NVFP4 has no supported decode action at M=16.
        if item['prefill_method'] == 'sparse_nvfp4':
            item['prefill_method'] = 'dense_nvfp4'; repaired += 1
        if item['decode_method'] == 'sparse_nvfp4':
            item['decode_method'] = 'dense_nvfp4'; repaired += 1
        pair = item['prefill_method'], item['decode_method']
        if pair not in LEGAL:
            item['decode_method'] = item['prefill_method']; repaired += 1
    policy['policy_kind'] = 'speed_calibration_legal_projection'
    out = ROOT / 'speed_calibration_util085/policies'; out.mkdir(parents=True, exist_ok=True)
    (out / f'{a.policy_id}.json').write_text(json.dumps(policy, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'policy_id': a.policy_id, 'repaired_modules': repaired, 'output': str(out / f'{a.policy_id}.json')}))

if __name__ == '__main__': main()
