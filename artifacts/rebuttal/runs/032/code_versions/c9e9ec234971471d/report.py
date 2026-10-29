"""Full task and speed sensitivity; no winner selected from benchmark scores."""
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

from common import RUN, write_json
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
    for path, digest in design['source_sha256'].items():
        assert sha(Path(path)) == digest, path
    for path, digest in design['canonical_sha256'].items():
        with Path(path).open('rb') as handle:
            assert hashlib.file_digest(handle, 'sha256').hexdigest() == digest
    for name, key in [('policy_registry', 'registry_sha256'), ('points', 'points_sha256'), ('selected_policies', 'selected_sha256')]:
        assert sha(RUN / f'configs/{name}.json') == design[key]
    registry = read(RUN / 'configs/policy_registry.json')
    for pid, spec in registry.items():
        assert sha(RUN / f'configs/policies/{pid}.json') == spec['policy_sha256']
    selected = read(RUN / 'configs/selected_policies.json')
    base = read(RUN / 'results/confirmation/uniform_dense_bf16/prefill_decode.json')['blocks']
    nll = {}
    for pid in registry:
        path = RUN / f'results/confirmation/{pid}/prefill_decode.json'
        value = read(path)
        assert len(value['blocks']) == 100 and value['data_sha256'] == design['dataset']['sha256']
        assert value['data_role'] == 'fixed_benchmark_test' and path.stat().st_mtime >= selected['frozen_at_unix']
        ds = [a['avg_nll'] - b['avg_nll'] for a, b in zip(value['blocks'], base, strict=True)]
        nll[pid] = guard(ds, pid == 'uniform_dense_bf16')
    save('benchmark_nll.csv', [{'policy': p, **v} for p, v in nll.items()])
    campaigns, per_policy = [], {}
    rounds = read(RUN / 'configs/formal_rounds.json')
    assert rounds['selected_sha256'] == design['selected_sha256']
    for i, order in enumerate(rounds['rounds']):
        records = {pid: read(RUN / f'results/measurements/formal_round{i}_{pid}.json') for pid in order}
        anchor = statistics.mean(records[p]['cases'][0]['medians']['e2e_ms'] for p in [order[0], order[-1]])
        int8 = records['uniform_int8']['cases'][0]['medians']['e2e_ms']
        for pid, record in records.items():
            assert record['input_sha256'] == design['speed_formal_sha256']
            assert record['placement']['cpu_affinity'] == list(range(8))
            assert all(v['exact_equal'] for v in record['exact_weight_audit'].values())
            assert sum(not s['warmup'] for s in record['cases'][0]['samples']) == 5
            if pid.endswith('_bracket_end'):
                continue
            medians = record['cases'][0]['medians']
            row = {'policy': pid, 'round': i, **medians, 'speedup_bf16': anchor / medians['e2e_ms'],
                   'speedup_int8': int8 / medians['e2e_ms']}
            campaigns.append(row)
            per_policy.setdefault(pid, []).append(row)
    save('formal_campaigns.csv', campaigns)
    speed = {pid: {k: statistics.mean(r[k] for r in rows) for k in rows[0] if k not in ['policy', 'round']}
             for pid, rows in per_policy.items()}
    save('formal_speed.csv', [{'policy': p, **v} for p, v in speed.items()])
    generation = read(RUN / 'configs/generation_jobs.json')
    tasks, origins = {}, {}
    for item in generation['reused']:
        assert sha(Path(item['source'])) == item['sha256']
        tasks[item['policy'], item['dataset']] = read(Path(item['source']))
        origins.setdefault(item['policy'], set()).add(item['source_run'] + ' same-map Graph')
    for item in generation['jobs']:
        tasks[item['policy'], item['dataset']] = read(RUN / f'results/generation/{item["policy"]}/{item["dataset"]}/metrics_0000_end.json')
        origins.setdefault(item['policy'], set()).add('032 new Graph')
    gen = {}
    for pid in registry:
        row = {}
        for dataset, key, metric, count in [('cnn_dm_1000', 'cnn_dm', 'rougeL_percent', 1000),
                ('dsum', 'dialogsum', 'rougeL_percent', 1500), ('IWSLT', 'iwslt', 'sacre_bleu', 333)]:
            value = tasks[pid, dataset]
            assert value['samples'] == count and value['input_sha256'] == sha(RUN / f'data/generation_{dataset}.jsonl')
            row[key] = value[metric]
        gen[pid] = {**row, 'generation_avg': statistics.mean(row.values()), 'source': ', '.join(sorted(origins[pid]))}
    save('full_generation.csv', [{'policy': p, **v} for p, v in gen.items()])
    mc_design = read(RUN / 'configs/mc_jobs.json')
    mc = {}
    for pid, source in mc_design['prefill_equivalent_source'].items():
        target_map = read(RUN / f'configs/policies/{pid}.json')['method_map']
        source_map = read(RUN / f'configs/policies/{source}.json')['method_map']
        assert all(target_map[n]['prefill_method'] == source_map[n]['prefill_method'] for n in target_map)
        row = {}
        for task, key, metric in [('arc_challenge', 'arc_c', 'acc_norm'), ('arc_easy', 'arc_e', 'acc'),
                                  ('winogrande', 'wino', 'acc'), ('mmlu', 'mmlu', 'acc')]:
            value = read(RUN / f'results/tasks/{source}/{task}/full/result.json')
            assert value['limit'] is None and value['num_fewshot'] == 0
            matches = [v for k, v in value['metrics'].items() if k.split(',')[0] == metric]
            assert len(matches) == 1, (task, value['metrics'])
            row[key] = matches[0] * 100
        mc[pid] = {**row, 'mc_avg': statistics.mean(row.values()), 'prefill_source': source}
    save('full_mc.csv', [{'policy': p, **v} for p, v in mc.items()])
    counts = {}
    for pid in registry:
        mapping = read(RUN / f'configs/policies/{pid}.json')['method_map']
        counts[pid] = {p: dict(Counter(v[p + '_method'] for v in mapping.values())) for p in ['prefill', 'decode']}
    write_json(RUN / 'results/policy_counts.json', counts)
    rows = []
    points = read(RUN / 'configs/points.json')
    for point in points:
        pid = point['policy']
        rows.append({'label': point['label'], 'policy': pid, 'prefill_weight': point['prefill_weight'], 'budget': point['budget'],
            'weighted_prediction': point['weighted_score'], 'original_prediction': point['original_prediction'],
            'sparse_fp4_prefill': counts[pid]['prefill'].get('sparse_nvfp4', 0),
            **nll[pid], 'mean_within_nominal_budget': nll[pid]['delta_nll'] <= point['budget'],
            'guard_within_nominal_budget': nll[pid]['guarded_delta_nll'] <= point['budget'],
            **speed[pid], **{k:v for k,v in gen[pid].items() if k!='source'}, **mc[pid]})
    save('sweep_results.csv', rows)
    baseline_mc, baseline_gen = mc['uniform_int8']['mc_avg'], gen['uniform_int8']['generation_avg']
    comparison = [{'label': r['label'], 'policy': r['policy'], 'speedup_int8': r['speedup_int8'],
        'mc_avg_delta_int8': r['mc_avg'] - baseline_mc, 'generation_avg_delta_int8': r['generation_avg'] - baseline_gen,
        'all_generation_parts_ge_int8': all(gen[r['policy']][k] >= gen['uniform_int8'][k] for k in ['cnn_dm', 'dialogsum', 'iwslt']),
        'all_mc_parts_ge_int8': all(mc[r['policy']][k] >= mc['uniform_int8'][k] for k in ['arc_c', 'arc_e', 'wino', 'mmlu'])} for r in rows]
    save('relative_to_int8.csv', comparison)
    display = [(p['label'], p['policy'], p['prefill_weight'], p['budget']) for p in points]
    display += [(p, p, None, None) for p in design['anchors']]
    lines = ['# Qwen3完整空间：预算与prefill误差权重', '',
        '固定016八候选Qwen代理，BF16零锚定；144模块独立选择64个阶段组合。λ只调整prefill质量代价，不禁止格式。先最小延迟、再同延迟最小预测质量损失。', '',
        '全部预算/权重在任务结果之前冻结：λ=1的ε=.01/.02/.03/.04，λ=1.5/2的ε=.03。λ≠1的分数是偏好得分，不是重新校准后的ΔNLL；不能仅凭同名ε认定不同λ等质量。比较完整精度—速度折中。', '',
        'λ=2/ε=.03与λ=1/ε=.02是同一有效map，复用同份测量，不算独立结果。完整生成Avg为两个ROUGE-L百分分数与BLEU的算术摘要；MC Avg为四项准确率等权均值。', '',
        '## 完整生成与三轮速度', '',
        '| Label | λ | ε | Prefill sparse FP4 | CNN/DM | DialogSum | IWSLT | Avg | E2E ms | ×INT8 | ×BF16 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for label, pid, weight, budget in display:
        lines.append(f'| {label} | {weight if weight is not None else "—"} | {budget if budget is not None else "—"} | '
            f'{counts[pid]["prefill"].get("sparse_nvfp4",0)} | {gen[pid]["cnn_dm"]:.3f} | {gen[pid]["dialogsum"]:.3f} | '
            f'{gen[pid]["iwslt"]:.3f} | {gen[pid]["generation_avg"]:.3f} | {speed[pid]["e2e_ms"]:.2f} | '
            f'{speed[pid]["speedup_int8"]:.3f} | {speed[pid]["speedup_bf16"]:.3f} |')
    lines += ['', '## 完整prefill MC精度', '', '| Label | ARC-C | ARC-E | Wino | MMLU | Avg acc |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for label, pid, _, _ in display:
        lines.append('| ' + label + ' | ' + ' | '.join(f'{mc[pid][k]:.3f}' for k in ['arc_c', 'arc_e', 'wino', 'mmlu', 'mc_avg']) + ' |')
    lines += ['', 'MC按相同prefill map共享新测结果：每批只做prompt likelihood，不使用decode权重生成后续token；四任务完整、0-shot、batch4、eager/3GiB KV，沿原正式协议。生成同map/源码/输入协议的027/028/029带SHA复用，来源单列于full_generation.csv。', '',
        'NLL为既有WikiText test固定100窗口，未称为全新盲测/独立确认；全部点评测，不依据NLL或最终任务分数挑λ/重选赢家。加权得分、原预测与实际NLL分别列于sweep_results.csv；保护值仍mean+2SE+.005，只作预定名义预算的诊断。', '',
        '速度GPU0/NUMA0，B8/S2048/O64，6GiB KV/native decode Graph；三轮3warmup+5timed，首尾BF16。E2E含prefill→decode实际切换，不包含下一批恢复；恢复和cycle另存。MC表没有把P+D TTFT冒充额外测过的prefill-only速度。', '',
        '[全部分项](results/sweep_results.csv) · [INT8相对比较](results/relative_to_int8.csv) · [三轮速度](results/formal_campaigns.csv) · [完整生成](results/full_generation.csv) · [MC](results/full_mc.csv) · [NLL](results/benchmark_nll.csv) · [配置分布](results/policy_counts.json)', '']
    (RUN / 'REPORT.md').write_text('\n'.join(lines))
    assert not list((RUN / 'data/checkpoints').rglob('*.safetensors'))
    evidence = [*list((RUN/'results/confirmation').glob('*/*.json')), *list((RUN/'results/measurements').glob('*.json')),
                *list((RUN/'results/generation').glob('*/*/metrics_*.json')), *list((RUN/'results/tasks').glob('*/*/full/result.json'))]
    write_json(RUN / 'results/final_audit.json', {'complete': True, 'verified': True, 'unique_maps': len(registry),
        'labels': len(points), 'new_generation_jobs': len(generation['jobs']), 'reused_generation_jobs': len(generation['reused']),
        'mc_unique_prefill_maps': len(mc_design['jobs']), 'code_sha256': {p.name: sha(p) for p in (RUN/'code').glob('*.py')},
        'evidence_sha256': {str(p.relative_to(RUN)): sha(p) for p in evidence},
        'csv_sha256': {p.name: sha(p) for p in (RUN/'results').glob('*.csv')},
        'retained_failures': [p.name for p in (RUN/'results').glob('failure*.json')]})
    print('Audited budget and weight sweep', comparison, flush=True)


if __name__ == '__main__':
    main()
