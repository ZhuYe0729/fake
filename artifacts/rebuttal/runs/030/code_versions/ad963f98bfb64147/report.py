"""Report guard forecasting and same-budget mean-only/cumulative comparisons."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

import torch

from common import REPO, RUN
from daso.modeling.quality_constraint import forecast_quality_guard
from selection import guard

read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def save(name, rows):
    with (RUN / 'results' / name).open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    design = read(RUN / 'configs/design.json')
    selected = read(RUN / 'configs/selected_policies.json')
    pool = read(RUN / 'configs/proposal_pool.json')
    registry = read(RUN / 'configs/policy_registry.json')
    for path, digest in {**design['source_sha256'], **design['calibration_reuse_sha256']}.items():
        assert sha(Path(path)) == digest, path
    for key, path in [('design_sha256', 'configs/design.json'), ('pool_sha256', 'configs/proposal_pool.json'),
            ('registry_sha256', 'configs/policy_registry.json'), ('validation_sha256', 'results/validation_candidates.json')]:
        assert selected[key] == sha(RUN / path), key
    assert pool['model_choice_sha256'] == sha(RUN / 'configs/model_choice.json')
    assert set(pool['families']['mean_only']) <= set(pool['families']['guard_aware'])
    for pid, value in registry.items():
        assert sha(RUN / f'configs/policies/{pid}.json') == value['policy_sha256']
    for pid, digest in selected['calibration_sha256'].items():
        assert sha(RUN / f'results/measurements/calibration_{pid}.json') == digest
    source = RUN.parent / '029'
    old = read(source / 'configs/design.json')

    def nll(pid):
        return torch.tensor([r['avg_nll'] for r in read(source / f'calibration/results/validation/{pid}/prefill_decode.json')['blocks']], dtype=torch.float64)

    base, reference = nll('uniform_sq_int8'), nll('uniform_dense_bf16')
    deltas = torch.stack([nll(p) - base for p in old['singles']])
    predicted_rows = []
    margin = read(RUN / 'results/guard_forecast.json')['residual_margin']
    for row in read(RUN / 'results/validation_candidates.json'):
        pid = row['policy']
        mask = registry[pid]['mask']
        pred = {'predicted_mean': None, 'predicted_2se': None, 'predicted_guard': None}
        if mask is not None:
            decision = torch.tensor([(mask >> i) & 1 for i in range(16)])
            forecast = forecast_quality_guard(base - reference, deltas, decision,
                validation_windows=128, residual_margin=margin)
            pred = {'predicted_mean': float(forecast.mean), 'predicted_2se': 2 * float(forecast.standard_error),
                    'predicted_guard': float(forecast.guarded)}
        predicted_rows.append({'policy': pid, 'w4_decode_modules': mask.bit_count() * 9 if mask is not None else 0,
            **pred, **row, 'feasible_e001': row['guarded_delta_nll'] <= .01,
            'feasible_e004': row['guarded_delta_nll'] <= .04})
    save('forecast_vs_validation.csv', predicted_rows)
    baseline = read(RUN / 'results/confirmation/uniform_dense_bf16/prefill_decode.json')
    confirmation = {}
    for path in (RUN / 'results/confirmation').glob('*/prefill_decode.json'):
        value = read(path)
        assert len(value['blocks']) == 128 and value['data_role'] == 'confirmation'
        assert value['data_sha256'] == design['datasets']['confirmation']['sha256']
        assert path.stat().st_mtime >= selected['frozen_at_unix']
        ds = [a['avg_nll'] - b['avg_nll'] for a, b in zip(value['blocks'], baseline['blocks'], strict=True)]
        confirmation[path.parent.name] = guard(ds, path.parent.name == 'uniform_dense_bf16')
    save('confirmation.csv', [{'policy': p, **c} for p, c in sorted(confirmation.items())])

    rounds = read(RUN / 'configs/formal_rounds.json')
    assert rounds['selected_sha256'] == sha(RUN / 'configs/selected_policies.json')
    campaigns, per_policy = [], {}
    for i, order in enumerate(rounds['rounds']):
        records = {pid: read(RUN / f'results/measurements/formal_round{i}_{pid}.json') for pid in order}
        anchors = [records[p]['cases'][0]['medians'] for p in [order[0], order[-1]]]
        anchor = {k: statistics.mean(a[k] for a in anchors) for k in ['e2e_ms', 'cycle_ms']}
        int8 = records['uniform_int8']['cases'][0]['medians']
        for pid, record in records.items():
            assert record['input_sha256'] == design['datasets']['speed_formal']['sha256']
            assert record['placement']['cpu_affinity'] == list(range(8))
            assert all(a['exact_equal'] for a in record['exact_weight_audit'].values())
            assert len([s for s in record['cases'][0]['samples'] if not s['warmup']]) == 5
            if pid.endswith('_bracket_end'):
                continue
            medians = record['cases'][0]['medians']
            row = {'policy': pid, 'campaign': i, **medians,
                'e2e_speedup': anchor['e2e_ms'] / medians['e2e_ms'],
                'cycle_speedup': anchor['cycle_ms'] / medians['cycle_ms'],
                'relative_e2e_to_int8': int8['e2e_ms'] / medians['e2e_ms'],
                'relative_cycle_to_int8': int8['cycle_ms'] / medians['cycle_ms']}
            campaigns.append(row)
            per_policy.setdefault(pid, []).append(row)
    save('formal_campaigns.csv', campaigns)
    speed = {}
    for pid, rows in per_policy.items():
        assert len(rows) == 3
        speed[pid] = {k: statistics.mean(r[k] for r in rows) for k in ['e2e_ms', 'cycle_ms', 'ttft_ms',
            'tpot_ms', 'restore_ms', 'e2e_speedup', 'cycle_speedup', 'relative_e2e_to_int8', 'relative_cycle_to_int8']}
    save('formal_speed.csv', [{'policy': p, **r} for p, r in speed.items()])
    comparisons = read(RUN / 'configs/model_selections.json')['choices']
    comparison_rows = []
    for c in comparisons:
        pid, budget = c['policy'], c['quality_budget']
        value = confirmation.get(pid)
        comparison_rows.append({'method': c['model'], 'budget': budget, 'policy': pid,
            'validation_guard': c['validation_and_calibration']['guarded_delta_nll'],
            'confirmation_delta_nll': value['delta_nll'] if value else None,
            'confirmation_guard': value['guarded_delta_nll'] if value else None,
            'confirmation_mean_pass': value['delta_nll'] <= budget if value else None,
            'confirmation_guard_pass': value['guarded_delta_nll'] <= budget if value else None,
            **speed[pid]})
    save('same_budget_comparison.csv', comparison_rows)
    primary = [r for r in comparison_rows if r['method'] == 'guard_aware']
    save('selected_results.csv', primary)
    generation = read(RUN / 'configs/generation_jobs.json')
    assert generation['selected_sha256'] == sha(RUN / 'configs/selected_policies.json')
    tasks = {}
    for item in generation['reused']:
        assert sha(Path(item['source'])) == item['sha256']
        tasks[item['policy'], item['dataset']] = (item['metric'], item['source_run'] + ' same-map Graph')
    for job in generation['jobs']:
        path = RUN / f'results/generation/{job["policy"]}/{job["dataset"]}/metrics_0000_end.json'
        tasks[job['policy'], job['dataset']] = (read(path), '030 new Graph')
    genrows = []
    for pid in sorted({p for p, _ in tasks}):
        row, sources = {'policy': pid}, set()
        for dataset, key, metric, count in [('cnn_dm_1000', 'cnn_dm', 'rougeL_percent', 1000),
                ('dsum', 'dialogsum', 'rougeL_percent', 1500), ('IWSLT', 'iwslt', 'sacre_bleu', 333)]:
            value, origin = tasks[pid, dataset]
            assert value['samples'] == count and value['input_sha256'] == sha(RUN / f'data/generation_{dataset}.jsonl')
            row[key] = value[metric]
            sources.add(origin)
        row['avg_score'] = statistics.mean(row[k] for k in ['cnn_dm', 'dialogsum', 'iwslt'])
        row['source'] = ', '.join(sorted(sources))
        genrows.append(row)
    save('full_generation.csv', genrows)

    lines = ['# Qwen3：考虑验证保护值的条件增量提案', '',
        '030复用029条件加和模型及训练64窗口，不新增模型校准GPU作业。029开发validation显示仅优化均值的点可能被最终SE/余量筛掉，因此本轮在提案生成中预测128窗口保护值。使用配对anchor/组增量完整协方差，另加8个校准policy holdout的最大正残差；不是形式化置信保证。', '',
        f'固定残差余量={margin:.6f}，提案guard目标为.005/.01/.02/.04，枚举65536组mask。最终池{len(registry)}个map，保留029全部条件均值提案和共同anchor；累计池不会人为丢掉mean-only的可行incumbent。模型/边际/协方差/余量未使用029 confirmation或030最终窗口重新拟合。', '',
        '本轮128validation+128confirmation+16速度输入均为unused内部WikiText train窗口，排除原训练128、028的208和029的272窗口。不是全新官方语料。选择在confirmation前冻结，同mean+2SE+.005规则，未放宽ε。', '',
        '## 同预算、同数据、同运行时比较', '',
        '| Method | ε | Policy | Val guard | Confirm ΔNLL | Confirm guard | Mean pass | Guard pass | E2E × | Cycle × | Relative E2E /INT8 |',
        '| --- | ---: | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |']
    for r in comparison_rows:
        fmt = lambda x: f'{x:.5f}' if x is not None else '未确认'
        lines.append(f'| {r["method"]} | {r["budget"]:.2f} | {r["policy"]} | {r["validation_guard"]:.5f} | '
            f'{fmt(r["confirmation_delta_nll"])} | {fmt(r["confirmation_guard"])} | {r["confirmation_mean_pass"]} | '
            f'{r["confirmation_guard_pass"]} | {r["e2e_speedup"]:.3f} | {r["cycle_speedup"]:.3f} | {r["relative_e2e_to_int8"]:.3f} |')
    lines += ['', '只对主版本和固定anchor开独立确认；mean-only若未与这些map重合，确认列留空，不冒充新测。两种方法的选择均在相同新validation上按真实guard和calibration E2E完成。mean-only选择也进入正式测速控制。GPU0/NUMA0、B8/S2048/O64、KV6GiB、2120/16384/[8]、3轮首尾BF16锚点，每轮3warmup+5timed，权重切换逐tensor相等检查通过。cycle含恢复，原始三轮保留，不把小差值称为显著。', '',
        '## 完整生成', '', '| Policy | CNN/DM | DialogSum | IWSLT | Avg score | Source |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for r in genrows:
        lines.append('| ' + r['policy'] + ' | ' + ' | '.join(f'{r[k]:.3f}' for k in ['cnn_dm', 'dialogsum', 'iwslt', 'avg_score']) + ' | ' + r['source'] + ' |')
    lines += ['', '完整1000/1500/333是已有benchmark上的冻结策略评测，不反向选择配置；同map/代码/数据协议的027/028 Graph分数明确继承，不混入旧eager。Avg是ROUGE-L%/BLEU算术摘要，不是NLL约束。本轮未重跑四项MC。', '',
        '## 边界', '',
        '- 经验残差只有8个policy holdout，协方差只有64个训练窗口；保护值预测仍可能失准，必须看新confirmation的均值与保护值，不宣称任意分布保证。',
        '- 仅固定SQ-prefill的16组子空间，不是全144模块/全部8候选全局最优。旧结果、失败和均值提案均保留。',
        '- 不读取confirmation来改模型、窗口数、预算或重选；不将相同map的多预算标签计为独立策略。', '',
        '[预测guard与真实validation](results/forecast_vs_validation.csv) · [同预算比较](results/same_budget_comparison.csv) · [三轮速度](results/formal_speed.csv) · [完整生成](results/full_generation.csv) · [残差来源](results/guard_forecast.json)', '']
    (RUN / 'REPORT.md').write_text('\n'.join(lines))
    evidence = [*list((RUN / 'results/validation').glob('*/*.json')), *list((RUN / 'results/confirmation').glob('*/*.json')),
        *list((RUN / 'results/measurements').glob('*.json')), *list((RUN / 'results/generation').glob('*/*/metrics_*.json'))]
    assert not list((RUN / 'data/checkpoints').glob('*/model.safetensors'))
    audit = {'complete': True, 'verified': True, 'validation_maps': len(registry),
        'confirmation_maps': len(confirmation), 'new_calibration_jobs': 0,
        'new_full_generation_jobs': len(generation['jobs']), 'reused_full_generation_jobs': len(generation['reused']),
        'design_sha256': sha(RUN / 'configs/design.json'), 'selected_sha256': sha(RUN / 'configs/selected_policies.json'),
        'source_calibration_unchanged': True, 'selection_before_confirmation': True,
        'primary_mean_pass': all(r['confirmation_mean_pass'] for r in primary),
        'primary_guard_pass': all(r['confirmation_guard_pass'] for r in primary),
        'code_sha256': {p.name: sha(p) for p in (RUN / 'code').glob('*.py')},
        'evidence_sha256': {str(p.relative_to(RUN)): sha(p) for p in evidence},
        'csv_sha256': {p.name: sha(p) for p in (RUN / 'results').glob('*.csv')}}
    (RUN / 'results/final_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print('Audited guard-aware study', primary, flush=True)


if __name__ == '__main__':
    main()
