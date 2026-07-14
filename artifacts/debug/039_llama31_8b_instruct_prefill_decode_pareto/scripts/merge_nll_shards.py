#!/usr/bin/env python3
"""Merge only valid one-row NLL shards; fail if the frozen suite is incomplete."""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest = json.loads((ROOT / 'policies/prefill_decode/manifest.json').read_text())
    rows, missing = [], []
    for item in manifest:
        path = ROOT / 'nll_shards' / f"{item['policy_id']}.csv"
        if not path.exists(): missing.append(item['policy_id']); continue
        values = list(csv.DictReader(path.open()))
        if len(values) != 1 or values[0].get('policy_id') != item['policy_id']:
            raise ValueError(f'invalid shard: {path}')
        rows.append({**values[0], 'split': item['split'], 'policy_kind': item['policy_kind']})
    if missing: raise RuntimeError(f'missing {len(missing)} NLL shards: {missing}')
    out = ROOT / 'nll/prefill_decode.csv'; out.parent.mkdir(exist_ok=True)
    with out.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(out)

if __name__ == '__main__': main()
