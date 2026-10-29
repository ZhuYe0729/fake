"""Descriptive held-out frontiers; never used to choose or retune a policy."""
import csv
import json
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import RUN, write_json
from analyze_constraints import quality


def read(path):
    return json.loads(path.read_text())


def main():
    design = read(RUN / 'configs/design.json')
    controls = read(RUN / 'configs/hawq_latency_controls.json')['points']
    labels = {}
    for index, row in enumerate(design['rows']):
        family = ('Uniform' if index < 10 else 'HAWQ' if index < 13 else
                  'DASO DP' if index < 15 else 'DASO validation')
        for scenario in ['prefill_only', 'prefill_decode']:
            labels.setdefault((row[scenario]['policy'], scenario), []).append((row['method'], family))
    for p in controls:
        labels.setdefault((p['policy_id'], p['scenario']), []).append((p['label'], 'HAWQ'))
    rows = []
    for (pid, scenario), aliases in sorted(labels.items()):
        mean, se = quality(pid, scenario)
        output = 1 if scenario == 'prefill_only' else 64
        ratios, times = [], []
        for number in range(3):
            data = read(RUN / f'results/speed/node0/round{number}/{pid}.json')
            values = [s['e2e_ms'] for c in data['cases'] if c['output'] == output for s in c['samples'] if not s['warmup']]
            anchor = []
            for suffix in ['', '_bracket_end']:
                baseline = read(RUN / f'results/speed/node0/round{number}/uniform_dense_bf16{suffix}.json')
                anchor += [s['e2e_ms'] for c in baseline['cases'] if c['output'] == output for s in c['samples'] if not s['warmup']]
            if pid == 'uniform_dense_bf16':
                values = anchor
            times.append(statistics.median(values))
            ratios.append(statistics.median(anchor) / statistics.median(values))
        families = sorted({a[1] for a in aliases})
        rows.append({'policy': pid, 'scenario': scenario, 'test_delta_nll': mean, 'paired_se': se,
            'e2e_ms': statistics.median(times), 'speedup': statistics.median(ratios),
            'minimum_campaign_speedup': min(ratios), 'maximum_campaign_speedup': max(ratios),
            'families': ';'.join(families), 'labels': ';'.join(a[0] for a in aliases)})
    with (RUN / 'results/heldout_frontier_points.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    styles = {'Uniform': ('#666666', 'D'), 'HAWQ': ('#0072B2', 'o'),
              'DASO DP': ('#009E73', 's'), 'DASO validation': ('#D55E00', '*')}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), constrained_layout=True)
    for axis, scenario, title in zip(axes, ['prefill_only', 'prefill_decode'], ['Prefill only', 'Prefill + decode'], strict=True):
        for family, (color, marker) in styles.items():
            local = [r for r in rows if r['scenario'] == scenario and family in r['families'].split(';')]
            axis.errorbar([r['test_delta_nll'] for r in local], [r['speedup'] for r in local],
                xerr=[2*r['paired_se'] for r in local],
                yerr=[[r['speedup']-r['minimum_campaign_speedup'] for r in local],
                      [r['maximum_campaign_speedup']-r['speedup'] for r in local]],
                fmt=marker, color=color, markersize=8 if marker == '*' else 5,
                capsize=2, elinewidth=.7, alpha=.85, label=family)
        axis.axvline(0, color='black', linewidth=.6, alpha=.4)
        axis.axhline(1, color='black', linewidth=.6, alpha=.4)
        axis.set(title=title, xlabel='Held-out mean delta NLL (nats/token)', ylabel='E2E speedup vs BF16')
        axis.grid(alpha=.18)
    axes[0].legend(fontsize=8)
    for suffix in ['pdf', 'svg', 'png']:
        fig.savefig(RUN / f'results/heldout_frontier.{suffix}', dpi=180)
    plt.close(fig)
    write_json(RUN / 'results/frontier_scope.json', {'points': len(rows),
        'scope': 'Descriptive only. Independent test labels do not select configurations. Horizontal bars are paired +/-2SE over 64 windows; vertical bars are the min/max of three campaign speedups, not confidence intervals. A shared policy can appear in multiple method families; CSV preserves all aliases. No matched-quality dominance is inferred solely from the Hessian objective or nominal epsilon.'})


if __name__ == '__main__':
    main()
