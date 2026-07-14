#!/usr/bin/env python3
"""Solve phase-pair mixed policies under the frozen quality proxy.

This is a screening solver only.  Every selected policy is subsequently
exported and measured by vLLM and the WikiText evaluator.
"""
from __future__ import annotations
import csv, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('dense_bf16', 'dense_nvfp4', 'sparse_bf16', 'sparse_nvfp4', 'w4a16_ours')
KERNEL = {**{x: x for x in METHODS[:-1]}, 'w4a16_ours': 'marlin_nvfp4'}
TYPES = ('qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj')
LEGAL = {(x, x) for x in METHODS} | {('dense_nvfp4', 'w4a16_ours'), ('w4a16_ours', 'dense_nvfp4')}
STEP = 0.01

def rows(path):
    with path.open(newline='') as f: return list(csv.DictReader(f))


def pava(values):
    blocks = []
    for value in values:
        blocks.append([value, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            right = blocks.pop()
            blocks[-1][0] += right[0]
            blocks[-1][1] += right[1]
    return [total / count for total, count in blocks for _ in range(int(count))]


def load_speed_calibrator():
    path = ROOT / 'speed_calibration_continuous085_w6' / 'calibration.csv'
    if not path.exists():
        return None
    train = sorted((float(r['raw_predicted_linear_ms']), float(r['e2e_median_ms']))
                   for r in rows(path) if r['split'] == 'train')
    xs = [x for x, _ in train]
    ys = pava([y for _, y in train])

    def predict(raw):
        if raw <= xs[0]:
            return ys[0]
        if raw >= xs[-1]:
            return ys[-1]
        for index in range(1, len(xs)):
            if raw <= xs[index]:
                left, right = xs[index - 1], xs[index]
                return ys[index - 1] + (raw - left) / (right - left) * (ys[index] - ys[index - 1])
        raise AssertionError('unreachable')
    return predict

def main():
    model = json.loads((ROOT / 'reports/quality/model.json').read_text())
    speed_calibrator = load_speed_calibrator()
    coef = lambda mi, bucket, typ: math.log1p(math.exp(model['global'] + model['method'][mi] + model['bucket'][bucket] + model['type'][typ])) / model['feature_scale']
    errors = {}
    for phase in ('prefill', 'decode'):
        for bucket in range(4):
            for typ in TYPES: errors[phase, bucket, typ, 'dense_bf16'] = 0.0
        for method in METHODS[1:]:
            for row in rows(ROOT / 'local_errors' / f'{phase}_{method}.csv'):
                errors[phase, int(row['layer_bucket']), row['fused_type'], method] = float(row['output_rel_mse'])
    actions = {(r['phase'], r['module_name'], r['kernel']): float(r['latency_ms'])
               for r in rows(ROOT / 'action_support.csv') if r['supported'] == 'True'}
    groups = []
    for layer in range(32):
        for group, typ in (('self_attn', 'qkv_proj'), ('self_attn', 'o_proj'), ('mlp', 'gate_up_proj'), ('mlp', 'down_proj')):
            name = f'model.layers.{layer}.{group}.{typ}'; bucket = layer // 8; options = []
            for pre, dec in sorted(LEGAL):
                if ('prefill', name, KERNEL[pre]) not in actions or ('decode', name, KERNEL[dec]) not in actions: continue
                q = errors['prefill', bucket, typ, pre] * coef(METHODS.index(pre), bucket, TYPES.index(typ))
                q += 80 * errors['decode', bucket, typ, dec] * coef(METHODS.index(dec), bucket, TYPES.index(typ))
                # Keep non-dense choices visible to the integer DP.
                qb = 0 if q == 0 else max(1, math.ceil(q / STEP))
                latency = actions['prefill', name, KERNEL[pre]] + 80 * actions['decode', name, KERNEL[dec]]
                options.append((qb, q, latency, pre, dec))
            if not options: raise RuntimeError(f'no supported phase actions for {name}')
            groups.append((name, options))
    max_bin = 2500; dp = [math.inf] * (max_bin + 1); dp[0] = 0.0; back = []
    for _, options in groups:
        nxt = [math.inf] * (max_bin + 1); choice = [None] * (max_bin + 1)
        for old, cost in enumerate(dp):
            if not math.isfinite(cost): continue
            for oi, (qb, _, latency, _, _) in enumerate(options):
                new = min(max_bin, old + qb); candidate = cost + latency
                if candidate < nxt[new]: nxt[new] = candidate; choice[new] = (old, oi)
        dp = nxt; back.append(choice)
    budgets = (0, .05, .1, .2, .35, .5, .75, 1.0, 1.5, 2.0, 3., 4., 6., 8., 12., 16., 20.)
    out = ROOT / 'pareto'; policy_dir = out / 'policies'; policy_dir.mkdir(parents=True, exist_ok=True)
    result, seen = [], set()
    for budget in budgets:
        limit = min(max_bin, int(budget / STEP)); state = min(range(limit + 1), key=lambda x: dp[x])
        picks = []
        for index in range(len(groups) - 1, -1, -1):
            previous, option = back[index][state]; picks.append(option); state = previous
        picks.reverse(); signature = tuple(picks)
        if signature in seen: continue
        seen.add(signature); selected = [opts[pick] for (_, opts), pick in zip(groups, picks)]
        pid = f'point_{len(result):03d}'
        method_map = {name: {'prefill_method': selected[i][3], 'decode_method': selected[i][4]} for i, (name, _) in enumerate(groups)}
        (policy_dir / f'{pid}.json').write_text(json.dumps({'policy_id': pid, 'scenario': 'prefill_decode', 'policy_kind': 'predicted_pareto_screening', 'default_prefill_method': 'dense_bf16', 'default_decode_method': 'dense_bf16', 'modules_to_not_convert': ['lm_head'], 'method_map': method_map}, indent=2, sort_keys=True) + '\n')
        raw_speed = sum(x[2] for x in selected)
        result.append({'point_index': len(result), 'policy_id': pid, 'quality_budget': budget,
                       # The fitted intercept is a finite-sample nuisance: a
                       # dense BF16 policy must be the exact zero-delta anchor.
                       'predicted_delta_nll': sum(x[1] for x in selected),
                       'raw_predicted_linear_ms': raw_speed,
                       'calibrated_predicted_e2e_ms': speed_calibrator(raw_speed) if speed_calibrator else None,
                       **{f'prefill_count_{m}': sum(x[3] == m for x in selected) for m in METHODS},
                       **{f'decode_count_{m}': sum(x[4] == m for x in selected) for m in METHODS}})
    dense = next(r['raw_predicted_linear_ms'] for r in result if r['quality_budget'] == 0)
    dense_calibrated = next(r['calibrated_predicted_e2e_ms'] for r in result if r['quality_budget'] == 0)
    for row in result:
        row['raw_speedup_vs_dense'] = dense / row['raw_predicted_linear_ms']
        row['calibrated_speedup_vs_dense'] = (dense_calibrated / row['calibrated_predicted_e2e_ms']
                                               if row['calibrated_predicted_e2e_ms'] else None)
    with (out / 'predicted_points.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(result[0])); w.writeheader(); w.writerows(result)
    (out / 'metadata.json').write_text(json.dumps({'status': 'screening only; E2E/NLL closure required', 'quality': 'phase local+global; prefill + 80 decode', 'speed': 'supported kernel predictor sum + held-out monotone E2E calibration' if speed_calibrator else 'supported kernel predictor sum', 'quality_step': STEP, 'points': len(result)}, indent=2) + '\n')
    print(out / 'predicted_points.csv')

if __name__ == '__main__': main()
