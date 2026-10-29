"""Separate proxy generalization, budget confirmation and measured deployment."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from selection import guard

RUN = Path(__file__).resolve().parents[1]
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def save(name, rows):
    with (RUN / 'results' / name).open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def nll(role, pid):
    root = RUN / 'calibration' if role == 'calibration' else RUN
    mode = 'validation' if role == 'calibration' else role
    value = read(root / f'results/{mode}/{pid}/prefill_decode.json')
    return [b['avg_nll'] for b in value['blocks']]


def main():
    design = read(RUN / 'configs/design.json')
    choice = read(RUN / 'configs/model_choice.json')
    selected = read(RUN / 'configs/selected_policies.json')
    registry = read(RUN / 'configs/policy_registry.json')
    pool = read(RUN / 'configs/proposal_pool.json')
    for path, digest in {**design['source_sha256'], **design['protected_files']}.items():
        assert sha(Path(path)) == digest, path
    for name, digest in design['runtime_source_sha256'].items():
        assert sha(RUN / 'code' / name) == digest, name
    assert selected['design_sha256'] == sha(RUN / 'configs/design.json')
    assert selected['pool_sha256'] == sha(RUN / 'configs/proposal_pool.json')
    assert selected['registry_sha256'] == sha(RUN / 'configs/policy_registry.json')
    assert pool['model_choice_sha256'] == sha(RUN / 'configs/model_choice.json')
    assert selected['validation_sha256'] == sha(RUN / 'results/validation_candidates.json')
    for pid, digest in choice['calibration_sha256'].items():
        assert sha(RUN / f'calibration/results/validation/{pid}/prefill_decode.json') == digest
    for pid, digest in selected['calibration_sha256'].items():
        assert sha(RUN / f'results/measurements/calibration_{pid}.json') == digest
    assert choice['chosen_at_unix'] <= selected['frozen_at_unix']
    assert not set(design['fit_policies']) & set(design['holdouts'])
    for pid, entry in registry.items():
        assert sha(RUN / f'configs/policies/{pid}.json') == entry['policy_sha256']
    anchor = nll('calibration', 'uniform_sq_int8')
    group_rows = []
    singles = [nll('calibration', pid) for pid in design['singles']]
    for i, values in enumerate(singles):
        delta = [a - b for a, b in zip(values, anchor, strict=True)]
        group_rows.append({'group': i, 'type': design['groups'][i][0].rsplit('.', 1)[-1],
            'bucket': i // 4, 'modules': 9, 'delta_relative_to_sq': statistics.mean(delta),
            'paired_2se': 2 * statistics.stdev(delta) / math.sqrt(64)})
    save('group_increment_diagnostic.csv', group_rows)
    validation = read(RUN / 'results/validation_candidates.json')
    predictions = read(RUN / 'results/proposal_predictions.json')
    validation_rows = []
    for row in validation:
        validation_rows.append({**row, **{m: predictions.get(row['policy'], {}).get(m)
                                          for m in design['models']}})
    save('validation_candidates.csv', validation_rows)
    metrics = []
    for model in design['models']:
        errors = [r[model] - r['delta_nll'] for r in validation_rows if r[model] is not None]
        metrics.append({'model': model, 'maps': len(errors), 'mae': statistics.mean(abs(x) for x in errors),
            'rmse': math.sqrt(statistics.mean(x * x for x in errors)),
            'false_feasible_e001': sum(r[model] <= .01 and r['delta_nll'] > .01
                for r in validation_rows if r[model] is not None)})
    save('fresh_validation_prediction_metrics.csv', metrics)

    reference = nll('confirmation', 'uniform_dense_bf16')
    confirmation = {}
    for path in (RUN / 'results/confirmation').glob('*/prefill_decode.json'):
        result = read(path)
        assert result['data_role'] == 'confirmation' and len(result['blocks']) == 128
        assert result['data_sha256'] == design['datasets']['confirmation']['sha256']
        assert path.stat().st_mtime >= selected['frozen_at_unix']
        delta = [r['avg_nll'] - b for r, b in zip(result['blocks'], reference, strict=True)]
        confirmation[path.parent.name] = guard(delta, path.parent.name == 'uniform_dense_bf16')
    save('confirmation.csv', [{'policy': p, **v} for p, v in sorted(confirmation.items())])

    campaign_rows, grouped = [], {}
    rounds = read(RUN / 'configs/formal_rounds.json')
    assert rounds['selected_sha256'] == sha(RUN / 'configs/selected_policies.json')
    for i, order in enumerate(rounds['rounds']):
        measures = {pid: read(RUN / f'results/measurements/formal_round{i}_{pid}.json') for pid in order}
        start, end = [measures[p]['cases'][0]['medians'] for p in [order[0], order[-1]]]
        base = {k: (start[k] + end[k]) / 2 for k in ['e2e_ms', 'cycle_ms']}
        int8 = measures['uniform_int8']['cases'][0]['medians']
        for label, value in measures.items():
            assert value['input_sha256'] == design['datasets']['speed_formal']['sha256']
            assert value['placement']['cpu_affinity'] == list(range(8))
            assert all(x['exact_equal'] for x in value['exact_weight_audit'].values())
            assert len([x for x in value['cases'][0]['samples'] if not x['warmup']]) == 5
            if label.endswith('_bracket_end'):
                continue
            median = value['cases'][0]['medians']
            row = {'policy': label, 'campaign': i, **median,
                'e2e_speedup': base['e2e_ms'] / median['e2e_ms'],
                'cycle_speedup': base['cycle_ms'] / median['cycle_ms'],
                'relative_e2e_to_int8': int8['e2e_ms'] / median['e2e_ms'],
                'relative_cycle_to_int8': int8['cycle_ms'] / median['cycle_ms']}
            campaign_rows.append(row)
            grouped.setdefault(label, []).append(row)
    save('formal_campaigns.csv', campaign_rows)
    speed = {p: {key: statistics.mean(r[key] for r in rows) for key in [
        'e2e_ms', 'cycle_ms', 'ttft_ms', 'tpot_ms', 'restore_ms', 'e2e_speedup', 'cycle_speedup',
        'relative_e2e_to_int8', 'relative_cycle_to_int8']} for p, rows in grouped.items()}
    assert all(len(rows) == 3 for rows in grouped.values())
    save('formal_speed.csv', [{'policy': p, **v} for p, v in speed.items()])
    selection_rows, counts = [], {}
    for item in selected['selected']:
        pid, budget = item['policy'], item['quality_budget']
        mask = registry[pid]['mask']
        counts[pid] = {'mask': mask, 'w4_decode_modules': mask.bit_count() * 9 if mask is not None else 0,
                      'prefill_fixed_sq': mask is not None}
        c = confirmation[pid]
        selection_rows.append({'budget': budget, 'model': selected['primary_model'], 'policy': pid,
            'w4_decode_modules': counts[pid]['w4_decode_modules'],
            'validation_guard': item['validation_and_calibration']['guarded_delta_nll'],
            'confirmation_delta_nll': c['delta_nll'], 'confirmation_2se': 2 * c['paired_se'],
            'confirmation_guard': c['guarded_delta_nll'], 'mean_pass': c['delta_nll'] <= budget,
            'guard_pass': c['guarded_delta_nll'] <= budget, **speed[pid]})
    save('selected_results.csv', selection_rows)
    (RUN / 'results/selected_method_counts.json').write_text(json.dumps(counts, indent=2) + '\n')

    generation = read(RUN / 'configs/generation_jobs.json')
    assert generation['selected_sha256'] == sha(RUN / 'configs/selected_policies.json')
    tasks = {}
    for item in generation['reused']:
        assert sha(Path(item['source'])) == item['sha256']
        tasks[item['policy'], item['dataset']] = (item['metric'], item['source_run'] + ' same-map Graph')
    for job in generation['jobs']:
        path = RUN / f'results/generation/{job["policy"]}/{job["dataset"]}/metrics_0000_end.json'
        tasks[job['policy'], job['dataset']] = (read(path), '029 new Graph')
    genrows = []
    for pid in sorted({p for p, _ in tasks}):
        row, sources = {'policy': pid}, set()
        for dataset, key, metric, count in [('cnn_dm_1000', 'cnn_dm', 'rougeL_percent', 1000),
                ('dsum', 'dialogsum', 'rougeL_percent', 1500), ('IWSLT', 'iwslt', 'sacre_bleu', 333)]:
            data, source = tasks[pid, dataset]
            assert data['samples'] == count and data['input_sha256'] == sha(RUN / f'data/generation_{dataset}.jsonl')
            row[key] = data[metric]
            sources.add(source)
        row['avg_score'] = statistics.mean(row[k] for k in ['cnn_dm', 'dialogsum', 'iwslt'])
        row['source'] = ', '.join(sorted(sources))
        genrows.append(row)
    save('full_generation.csv', genrows)

    lines = ['# Qwen3：固定prefill的条件增量质量模型', '',
        '本轮为DASO可选质量代理分支：固定SQ-INT8 prefill，16个深度/类型组，每组9个Linear，decode组选择SQ-INT8或W4A16。相同真实Graph/KV路径采集配对整模型NLL，未改量化/kernel、预算或mean+2SE+.005门控。原质量模型与封存主表保留。', '',
        '校准采用原训练数据前64窗口；26个拟合policy与8个未拟合组合预先固定。校准policy holdout用于模型选择，不是新语料测试。128 validation/128 confirmation来自token-disjoint内部WikiText train窗口，排除原128训练与028全部208窗口；不是新官方benchmark语料。', '',
        '## 代理预测诊断', '',
        '| Model | Policy-holdout MAE | Policy-holdout RMSE | Fresh-validation MAE | Fresh-validation RMSE |',
        '| --- | ---: | ---: | ---: | ---: |']
    old_metrics = {m['model']: m for m in choice['metrics']}
    for m in metrics:
        old = old_metrics[m['model']]
        lines.append(f'| {m["model"]} | {old["mae"]:.5f} | {old["rmse"]:.5f} | {m["mae"]:.5f} | {m["rmse"]:.5f} |')
    lines += ['', f'主版本 **{selected["primary_model"]}** 仅由8个预定policy holdout的MAE选定，早于最终validation；不根据后者或confirmation反选模型。refit_original保留原架构/超参，使用相同26个新标签；old_anchored只纠正基线偏置。其余两模型直接校准signed分组增量，pair版额外包含8个预定交互。', '',
        '## 冻结主版本与独立确认', '',
        '| ε | Policy | W4 decode /144 | Val guard | Confirm ΔNLL ±2SE | Confirm guard | Mean pass | Guard pass | E2E × | Cycle × | Relative E2E /INT8 |',
        '| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |']
    for r in selection_rows:
        lines.append(f'| {r["budget"]:.2f} | {r["policy"]} | {r["w4_decode_modules"]} | {r["validation_guard"]:.5f} | '
            f'{r["confirmation_delta_nll"]:.5f} ± {r["confirmation_2se"]:.5f} | {r["confirmation_guard"]:.5f} | '
            f'{r["mean_pass"]} | {r["guard_pass"]} | {r["e2e_speedup"]:.3f} | {r["cycle_speedup"]:.3f} | {r["relative_e2e_to_int8"]:.3f} |')
    lines += ['', '确认均值与保护值分列；任何失败均不反向重选。正式速度GPU0/NUMA0、B8/S2048/O64、KV6GiB、2120/16384/[8]、三轮BF16首尾锚点、每轮3warmup+5timed，权重切换逐tensor相等检查通过。三轮均值/原始分项全部保留，未将微小数值差称作统计显著。cycle包含恢复，E2E延续原边界。', '',
        '## 完整生成', '', '| Policy | CNN/DM | DialogSum | IWSLT | Avg score | Source |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for r in genrows:
        lines.append('| ' + r['policy'] + ' | ' + ' | '.join(f'{r[k]:.3f}' for k in ['cnn_dm', 'dialogsum', 'iwslt', 'avg_score']) + ' | ' + r['source'] + ' |')
    lines += ['', '1000/1500/333完整benchmark只描述冻结选择，不参与训练/模型选择。相同map、代码和数据协议的027/028 Graph分数带SHA复用；普通INT8使用028当前Graph分数，不混入002旧eager。Avg为ROUGE-L%/BLEU混合算术摘要，不是质量约束；本轮未重跑四项MC。', '',
        '## 适用范围与成本', '',
        f'- 36个校准map×64窗口，其中26个用于拟合、8个用于模型选择，另外固定控制单列；新增标签成本不能省略。最终validation池为{len(pool["policies"])}个唯一map。',
        '- 各模型仅在共同16组子空间枚举65536个mask；目标为0/.005/.01/.04，不声称144模块全部8候选的全局最优。各模型的部署选点只使用本模型提案+共同anchor。',
        '- 配对交互是有限近似，不表达任意跨层传播；更低代理误差不自动等于更快部署或所有生成分数更高。可行INT8 anchor始终保留。',
        '- 新鲜validation数据只评估/选配置，confirmation只确认；旧结果不覆盖。所有参数、失败和不利结果保留。', '',
        '[逐组真实增量](results/group_increment_diagnostic.csv) · [policy holdout](results/policy_holdout_predictions.csv) · [新validation误差](results/fresh_validation_prediction_metrics.csv) · [选点](results/selected_results.csv) · [速度](results/formal_speed.csv) · [完整生成](results/full_generation.csv)', '']
    (RUN / 'REPORT.md').write_text('\n'.join(lines))
    evidence = [*list((RUN / 'calibration/results/validation').glob('*/*.json')),
        *list((RUN / 'results/validation').glob('*/*.json')), *list((RUN / 'results/confirmation').glob('*/*.json')),
        *list((RUN / 'results/measurements').glob('*.json')), *list((RUN / 'results/generation').glob('*/*/metrics_*.json'))]
    assert not list((RUN / 'data/checkpoints').glob('*/model.safetensors'))
    audit = {'complete': True, 'verified': True, 'primary_model': selected['primary_model'],
        'calibration_maps': 36, 'validation_maps': len(pool['policies']), 'confirmation_maps': len(confirmation),
        'new_full_generation_tasks': len(generation['jobs']), 'reused_full_generation_tasks': len(generation['reused']),
        'design_sha256': sha(RUN / 'configs/design.json'), 'selected_sha256': sha(RUN / 'configs/selected_policies.json'),
        'code_sha256': {p.name: sha(p) for p in (RUN / 'code').glob('*.py')},
        'evidence_sha256': {str(p.relative_to(RUN)): sha(p) for p in evidence},
        'csv_sha256': {p.name: sha(p) for p in (RUN / 'results').glob('*.csv')},
        'confirmation_mean_pass': all(r['mean_pass'] for r in selection_rows),
        'confirmation_guard_pass': all(r['guard_pass'] for r in selection_rows),
        'primary_model_before_validation': True, 'selection_before_confirmation': True,
        'old_results_and_runtime_unchanged': True, 'owned_weight_cache_released': True}
    (RUN / 'results/final_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print('Audited', selected['primary_model'], selection_rows, flush=True)


if __name__ == '__main__':
    main()
