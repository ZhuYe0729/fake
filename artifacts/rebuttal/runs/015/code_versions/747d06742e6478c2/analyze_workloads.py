"""Report all prespecified transfer loads and both campaign ranges."""
import csv
import statistics

from common import RUN
from launch import read


def main():
    config = read(RUN / 'configs/representative_workloads.json')
    rows = []
    lines = ['# 固定主配置的代表负载迁移', '', config['scope'], '',
        '以下使用标准负载已冻结的P+D配置；没有为新负载重新优化或提供新负载的质量保证。', '',
        '| Load | Method | TTFT × | TPOT × | E2E × | E2E campaign range | Restore ms |',
        '| --- | --- | ---: | ---: | ---: | --- | ---: |']
    for workload in config['workloads']:
        for spec in config['rows']:
            pid = spec['prefill_decode']['policy']
            measurements = []
            for number in range(2):
                prefix = f'workload_{workload["label"]}_round{number}_'
                data = read(RUN / f'results/measurements/{prefix}{pid}.json')
                assert data['input_sha256'] == workload['input_sha256']
                timed = [s for s in data['cases'][0]['samples'] if not s['warmup']]
                assert len(timed) == 5
                anchors = []
                for suffix in ['', '_bracket_end']:
                    baseline = read(RUN / f'results/measurements/{prefix}uniform_dense_bf16{suffix}.json')
                    anchors += [s for s in baseline['cases'][0]['samples'] if not s['warmup']]
                actual = anchors if pid == 'uniform_dense_bf16' else timed
                measured = {m: statistics.median(s[m] for s in actual)
                            for m in ['e2e_ms', 'ttft_ms', 'tpot_ms', 'restore_ms', 'cycle_ms']}
                ratios = {m + '_speedup': statistics.median(s[m] for s in anchors) / measured[m]
                          for m in ['e2e_ms', 'ttft_ms', 'tpot_ms', 'cycle_ms']}
                row = {'load': workload['label'], 'batch': workload['batch'], 'prompt': workload['prompt'],
                    'output': workload['output'], 'method': spec['method'], 'policy': pid,
                    'round': number, **measured, **ratios}
                rows.append(row)
                measurements.append(row)
            med = lambda m: statistics.median(x[m] for x in measurements)
            spread = [x['e2e_ms_speedup'] for x in measurements]
            lines.append(f'| {workload["label"]} | {spec["method"]} | {med("ttft_ms_speedup"):.2f} | '
                f'{med("tpot_ms_speedup"):.2f} | {med("e2e_ms_speedup"):.2f} | '
                f'{min(spread):.2f}–{max(spread):.2f} | {med("restore_ms"):.2f} |')
    with (RUN / 'results/representative_workloads.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (RUN / 'results/representative_workloads.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
