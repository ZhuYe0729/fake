"""Report the entire fixed pool, confirmation and both timing objectives."""
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

from selection import guard, select

RUN = Path(__file__).resolve().parents[1]
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def save(name, rows):
    with (RUN / 'results' / name).open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    design = read(RUN / 'configs/design.json')
    selection = read(RUN / 'configs/selected_policies.json')
    assert selection['design_sha256'] == sha(RUN / 'configs/design.json')
    assert selection['validation_sha256'] == sha(RUN / 'results/validation_candidates.json')
    for pid, digest in selection['calibration_hashes'].items():
        assert digest == sha(RUN / f'results/measurements/calibration_{pid}.json')
    for rid, digest in design['original_manifests'].items():
        assert digest == sha(RUN.parent / rid / 'results/manifest.json')
    for name, digest in design['runtime_source_hashes'].items():
        assert digest == sha(RUN / 'code' / name)
    for pid, spec in design['policies'].items():
        assert spec['policy_sha256'] == sha(RUN / f'configs/policies/{pid}.json')
    validation = read(RUN / 'results/validation_candidates.json')
    valrows = [{key: r[key] for key in ['policy', 'family', 'predicted_linear_ms', 'predicted_delta_nll',
        'delta_nll', 'paired_se', 'guarded_delta_nll']} | {
        'guard_32': r['sample_size_trace']['32']['guarded_delta_nll'],
        'guard_64': r['sample_size_trace']['64']['guarded_delta_nll'],
        'feasible_e001': r['guarded_delta_nll'] <= .01, 'feasible_e004': r['guarded_delta_nll'] <= .04} for r in validation]
    save('all_validation_candidates.csv', valrows)
    baseline = read(RUN / 'results/confirmation/uniform_dense_bf16/prefill_decode.json')
    confirmation = {}
    for path in (RUN / 'results/confirmation').glob('*/prefill_decode.json'):
        score = read(path)
        assert score['data_sha256'] == design['datasets']['confirmation']['sha256']
        assert path.stat().st_mtime >= selection['frozen_at_unix']
        differences = [a['avg_nll'] - b['avg_nll'] for a, b in zip(score['blocks'], baseline['blocks'], strict=True)]
        assert len(differences) == 64
        confirmation[path.parent.name] = guard(differences, path.parent.name == 'uniform_dense_bf16')
    save('confirmation.csv', [{'policy': p, **values} for p, values in sorted(confirmation.items())])
    rounds = read(RUN / 'configs/formal_rounds.json')
    assert rounds['selected_sha256'] == sha(RUN / 'configs/selected_policies.json')
    aggregate, per_round = {}, []
    for i, order in enumerate(rounds['rounds']):
        measures = {p: read(RUN / f'results/measurements/formal_round{i}_{p}.json') for p in order}
        start, end = [measures[p]['cases'][0]['medians'] for p in [order[0], order[-1]]]
        anchor = {m: (start[m] + end[m]) / 2 for m in ['e2e_ms', 'cycle_ms']}
        for label, measure in measures.items():
            assert measure['complete'] and measure['mode'] == 'arena_graph'
            assert measure['input_sha256'] == design['datasets']['speed_formal']['sha256']
            assert measure['placement']['cpu_affinity'] == list(range(8))
            assert len([s for s in measure['cases'][0]['samples'] if not s['warmup']]) == 5
            assert all(v['exact_equal'] for v in measure['exact_weight_audit'].values())
            if label.endswith('_bracket_end'):
                continue
            medians = measure['cases'][0]['medians']
            row = {'policy': label, 'campaign': i, **medians,
                'e2e_speedup': anchor['e2e_ms'] / medians['e2e_ms'],
                'cycle_speedup': anchor['cycle_ms'] / medians['cycle_ms'],
                'bf16_bracket_relative_drift': end['e2e_ms'] / start['e2e_ms'] - 1}
            per_round.append(row)
            aggregate.setdefault(label, []).append(row)
    save('formal_campaigns.csv', per_round)
    speed = {}
    for pid, rows in aggregate.items():
        assert len(rows) == 3
        speed[pid] = {key: statistics.mean(r[key] for r in rows)
            for key in ['e2e_ms', 'ttft_ms', 'tpot_ms', 'restore_ms', 'cycle_ms', 'e2e_speedup', 'cycle_speedup']}
    save('formal_speed.csv', [{'policy': p, **values} for p, values in speed.items()])
    table = []
    val = {r['policy']: r for r in validation}
    counts = {}
    for choice in selection['selected']:
        pid, budget = choice['policy'], choice['quality_budget']
        mapping = read(RUN / f'configs/policies/{pid}.json')['method_map']
        counts[pid] = {phase: dict(Counter(v[phase + '_method'] for v in mapping.values()))
                       for phase in ['prefill', 'decode']}
        table.append({'budget': budget, 'objective': choice['objective'], 'policy': pid,
            'validation_delta_nll': val[pid]['delta_nll'], 'validation_guard': val[pid]['guarded_delta_nll'],
            'confirmation_delta_nll': confirmation[pid]['delta_nll'],
            'confirmation_2se': 2 * confirmation[pid]['paired_se'],
            'confirmation_mean_pass': confirmation[pid]['delta_nll'] <= budget,
            'confirmation_guard_pass': confirmation[pid]['guarded_delta_nll'] <= budget, **speed[pid]})
    save('selected_results.csv', table)
    (RUN / 'results/selected_method_counts.json').write_text(json.dumps(counts, indent=2) + '\n')
    calibrated = []
    for pid in selection['calibration_hashes']:
        median = read(RUN / f'results/measurements/calibration_{pid}.json')['cases'][0]['medians']
        calibrated.append({**val[pid], **median})
    ablation = []
    for budget in [.01, .04]:
        for objective in ['e2e_ms', 'cycle_ms']:
            for scope in ['anchors_only', 'anchors_plus_local_repair']:
                pool = [r for r in calibrated if scope != 'anchors_only' or r['family'] != 'baseline-anchored decode repair']
                winner, number = select(pool, budget, objective)
                ablation.append({'budget': budget, 'objective': objective, 'pool': scope,
                    'policy': winner['policy'], 'calibration_ms': winner[objective], 'feasible_candidates': number})
    save('candidate_pool_ablation.csv', ablation)
    generation = read(RUN / 'configs/generation_jobs.json')
    assert generation['selected_sha256'] == sha(RUN / 'configs/selected_policies.json')
    metrics = {}
    for inherited in generation['reused']:
        assert sha(Path(inherited['source'])) == inherited['sha256']
        metrics[inherited['policy'], inherited['dataset']] = (inherited['metric'], '027 same-map Graph')
    for job in generation['jobs']:
        path = RUN / f'results/generation/{job["policy"]}/{job["dataset"]}/metrics_0000_end.json'
        metrics[job['policy'], job['dataset']] = (read(path), '028 new Graph')
    genrows = []
    for pid in sorted({p for p, _ in metrics}):
        row = {'policy': pid}
        sources = set()
        for dataset, key, metric_key, count in [('cnn_dm_1000', 'cnn_dm', 'rougeL_percent', 1000),
                ('dsum', 'dialogsum', 'rougeL_percent', 1500), ('IWSLT', 'iwslt', 'sacre_bleu', 333)]:
            metric, source = metrics[pid, dataset]
            assert metric['samples'] == count and metric['input_sha256'] == sha(RUN / f'data/generation_{dataset}.jsonl')
            row[key] = metric[metric_key]
            sources.add(source)
        row['generation_avg_score'] = statistics.mean(row[k] for k in ['cnn_dm', 'dialogsum', 'iwslt'])
        row['source'] = ', '.join(sorted(sources))
        genrows.append(row)
    save('full_generation.csv', genrows)
    old = read(RUN.parent / '015/results/validation_summary.json')
    root = [p for p in old if p['scenario'] == 'prefill_decode' and p['policy_id'] in
            ['uniform_sq_int8', 'graph_exact7_prefill_decode_q0.01', 'graph_phase_sq_int8_w4a16_ours']]
    pool = read(RUN.parent / '015/configs/reanalysis_pool.json')
    old8 = [p for p in read(RUN.parent / '016/results/validation_candidates.json')
            if p['scenario'] == 'prefill_decode' and p['policy'] in ['uniform_int8', 'uniform_sq_int8']]
    (RUN / 'reference/old_selection_diagnosis.json').write_text(json.dumps({
        'run015': root, 'run015_int8_pool_entry': pool['policies']['uniform_int8'], 'run016': old8}, indent=2) + '\n')
    lines = ['# Qwen3：基线保留与部分 W4A16 decode 选点修正', '',
        '本轮沿用原量化/runtime、冻结训练代理、ε及 mean+2SE+.005 规则，新增24个固定候选和更多内部留出验证窗口。128窗口决定选择；先冻结选择，再打开64窗口确认。新窗口排除了原128个训练/激活校准样本的任何token区间，与所有新split也不重叠；它们来自原WikiText train语料，不称为全新官方测试集。', '',
        '## 选择与独立确认', '',
        '| ε | 选点目标 | 选中配置 | Val guard | Confirm ΔNLL ±2SE | Confirm mean pass | E2E × | Cycle × |',
        '| ---: | --- | --- | ---: | ---: | --- | ---: | ---: |']
    for r in table:
        lines.append(f'| {r["budget"]:.2f} | {r["objective"]} | {r["policy"]} | {r["validation_guard"]:.5f} | '
            f'{r["confirmation_delta_nll"]:.5f} ± {r["confirmation_2se"]:.5f} | {r["confirmation_mean_pass"]} | '
            f'{r["e2e_speedup"]:.3f} | {r["cycle_speedup"]:.3f} |')
    lines += ['', 'E2E为原主目标；cycle是实验前另声明的含恢复目标，不混为同一种DASO。确认结果不反向重选。正式速度为GPU0/NUMA0、B8/S2048/O64、6GiB KV、native decode Graph，3轮，每轮3 warmup+5 timed，首尾BF16锚点平均，最终取三轮均值。真实权重切换逐tensor精确相等检查通过。', '',
        '## 同轮速度对照', '', '| 配置 | E2E × | Cycle × | TTFT ms | TPOT ms | Restore ms |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for pid, r in speed.items():
        lines.append(f'| {pid} | {r["e2e_speedup"]:.3f} | {r["cycle_speedup"]:.3f} | {r["ttft_ms"]:.2f} | {r["tpot_ms"]:.3f} | {r["restore_ms"]:.2f} |')
    lines += ['', '## 完整生成任务', '', '| 配置 | 来源 | CNN/DM | DialogSum | IWSLT | Avg score |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for r in genrows:
        lines.append('| ' + r['policy'] + ' | ' + r['source'] + ' | ' + ' | '.join(f'{r[k]:.3f}' for k in
            ['cnn_dm', 'dialogsum', 'iwslt', 'generation_avg_score']) + ' |')
    lines += ['', '完整1000/1500/333输入为已有benchmark；分数只描述冻结选点，不参与选择。Avg是ROUGE-L%和BLEU的算术摘要，不替代NLL约束。相同prefill的选择题质量可从原均匀配置单列引用，本轮未重跑四项MC。', '',
        '## 本次修正解决的内容', '',
        '- 显式保留普通INT8、SQ-INT8及相关阶段组合，加入从基线出发的嵌套decode局部候选。训练代理仅决定替换顺序，不提前排除质量可行的基线。',
        '- 只按固定128窗口规则筛选，32/64窗口保护值另列用于样本量敏感性；不挑最有利样本量。',
        '- 同一可行池内，用实测calibration E2E或cycle选最小者，断言不慢于可行均匀INT8 incumbent。该保证属于calibration候选池，不是所有工作负载/任务/新确认数据上的必胜保证。',
        '- 旧精确解已经选中过全W4A16但真实NLL失败；本轮不将该失败改写为ε=.01成功，不改旧主表。', '',
        '[全部24个候选](results/all_validation_candidates.csv) · [选择结果](results/selected_results.csv) · [三轮速度](results/formal_campaigns.csv) · [完整生成](results/full_generation.csv) · [旧选点原因](reference/old_selection_diagnosis.json)', '']
    (RUN / 'REPORT.md').write_text('\n'.join(lines))
    evidence = [*list((RUN / 'results/validation').glob('*/*.json')),
        *list((RUN / 'results/confirmation').glob('*/*.json')), *list((RUN / 'results/measurements').glob('*.json')),
        *list((RUN / 'results/generation').glob('*/*/metrics_*.json'))]
    audit = {'verified': True, 'complete': True, 'validation_policies': len(validation),
        'confirmation_policies': len(confirmation), 'selected_points': len(table),
        'design_sha256': sha(RUN / 'configs/design.json'), 'selection_sha256': sha(RUN / 'configs/selected_policies.json'),
        'code_sha256': {p.name: sha(p) for p in (RUN / 'code').glob('*.py')},
        'evidence_sha256': {str(p.relative_to(RUN)): sha(p) for p in evidence},
        'derived_csv_sha256': {p.name: sha(p) for p in (RUN / 'results').glob('*.csv')},
        'confirmation_after_selection': True, 'original_manifests_unchanged': True,
        'runtime_sources_unchanged': True, 'confirmation_mean_pass': all(r['confirmation_mean_pass'] for r in table)}
    (RUN / 'results/final_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print('Audited run028', json.dumps(table), flush=True)


if __name__ == '__main__':
    main()
