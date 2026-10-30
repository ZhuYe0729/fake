"""Measure the machine's idle footprint and separate it from active-process gaps."""
import csv
import json
import time

from common import RUN, write_json
from resources import snapshot


def main():
    target = RUN / 'configs/nvml_baseline.json'
    assert not target.exists()
    with (RUN / 'configs/profile_entry.csv').open() as stream:
        original = [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(stream)]
    initial_used = {int(row['index']): int(row['memory.used [MiB]'].split()[0]) * 1024**2 for row in original}
    samples = []
    while len(samples) < 3:
        rows = snapshot()
        if all(not row['processes'] and row['utilization_percent'] <= 5 and
               abs(row['allocated_bytes'] - initial_used[row['index']]) <= 128 * 1024**2 for row in rows):
            samples.append(rows)
        else:
            samples = []
        time.sleep(2)
    gpus = [{'index': index, 'idle_allocated_bytes': max(sample[index]['allocated_bytes'] for sample in samples),
             'nvml_reserved_bytes': max(sample[index]['reserved_bytes'] for sample in samples)} for index in range(2)]
    busy = json.loads((RUN / 'configs/nvml_accounting_precheck.json').read_text())
    gaps = [sample['gpus'][0]['attribution_gap_bytes'] for sample in busy['samples']
            if len(sample['gpus'][0]['processes']) == 1]
    assert len(gaps) >= 2 and max(gaps) - min(gaps) < 32 * 1024**2
    write_json(target, {'gpus': gpus, 'gpu0_single_process_gap_bytes': max(gaps),
        'idle_samples': samples, 'busy_precheck': 'configs/nvml_accounting_precheck.json',
        'scope': 'Measured fixed unattributed idle footprint and one-context accounting gap, distinct from NVMLv2 reserved. '
                 'Idle footprint must agree with initial empty-device observations within128MiB. Formal monitoring adds128MiB tolerance.'})
    print('NVML baseline calibrated', gpus, 'running gap MiB', max(gaps) / 1024**2, flush=True)


if __name__ == '__main__':
    main()
