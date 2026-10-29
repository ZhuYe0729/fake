"""Prespecified transfer workloads; keep selected maps fixed, without retuning."""
import hashlib
import json

import torch

from common import RUN, write_json


def main():
    destination = RUN / 'configs/representative_workloads.json'
    assert not destination.exists()
    design = json.loads((RUN / 'configs/design.json').read_text())
    # Uniform strong controls plus both HAWQ formats and all DASO versions.
    rows = [design['rows'][i] for i in [0, 2, 3, 7, 10, 11, 12, 13, 14, 15, 16]]
    policies = sorted({r['prefill_decode']['policy'] for r in rows})
    generator = torch.Generator().manual_seed(148015991)
    workloads = []
    for label, batch, prompt, output in [('single_b1_s2048_o128', 1, 2048, 128),
        ('decode_b8_s512_o128', 8, 512, 128), ('long_b8_s4096_o64', 8, 4096, 64)]:
        path = RUN / f'configs/{label}_inputs.json'
        write_json(path, torch.randint(100, 150000, (batch, prompt), generator=generator).tolist())
        workloads.append({'label': label, 'batch': batch, 'prompt': prompt, 'output': output,
            'input': str(path), 'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    write_json(destination, {'rows': rows, 'policies': policies, 'workloads': workloads,
        'rounds': [policies, policies[::-1]], 'warmups': 3, 'iterations': 5,
        'scope': 'Fixed standard-load P+D maps transferred to three representative loads. No per-load quality guarantee or test-based retuning. Same GPU0/NUMA0,6GiB KV,decode Graph,includes TTFT/TPOT/E2E/restore. Two reverse-order campaigns with BF16 brackets.'})
    print('Workloads frozen', len(workloads), 'policies', len(policies), flush=True)


if __name__ == '__main__':
    main()
