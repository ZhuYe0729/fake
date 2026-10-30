"""Independently audit complete evidence and render the three-family tables."""
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics as stats

import torch

from common import REPO, RUN, write_json

OUT = REPO / 'artifacts/rebuttal/responses/reviewer4/pro6000_qwen32b_run055'
SOURCES = {}


def read(path, signature=None):
    path = Path(path)
    signature_now = hashlib.sha256(path.read_bytes()).hexdigest()
    assert signature is None or signature_now == signature, path
    SOURCES[str(path.relative_to(REPO))] = signature_now
    return json.loads(path.read_text())


def paired(values, baseline):
    differences = [a['avg_nll'] - b['avg_nll'] for a, b in zip(values, baseline, strict=True)]
    assert len(differences) == 100
    return stats.mean(differences), stats.stdev(differences) / math.sqrt(100)


def summarize_speed(rounds):
    result = {}
    keys = rounds[0]['case'].keys()
    for metric in keys:
        values = [row['case'][metric] for row in rounds]
        ratios = [row['reference'][metric] / row['case'][metric] for row in rounds]
        result[metric] = {'milliseconds': stats.mean(values), 'round_medians_ms': values,
            'speedup': stats.mean(ratios), 'round_speedups': ratios, 'min_speedup': min(ratios),
            'max_speedup': max(ratios)}
    e2e = result['e2e_ms']['round_medians_ms']
    result['variation_flag'] = ((max(e2e) - min(e2e)) / stats.mean(e2e) > .05 or
                               any(row['bf16_drift'] > .05 for row in rounds))
    return result


def main():
    for name in ['test_complete', 'formal_speed_complete', 'all_downstream_complete']:
        assert read(RUN / f'results/{name}.json')['complete']
    formal = read(RUN / 'configs/formal_points.json')
    design = read(RUN / 'configs/design.json')
    rules = read(RUN / 'configs/selection_rules.json', formal['selection_rules_sha256'])
    assert rules['formal_speed_rounds'] == 3
    read(RUN / 'configs/search_extension.json')
    refinement = read(RUN / 'configs/low_budget_refinement.json')
    assert read(RUN / 'results/low_budget_validation_complete.json')['complete']
    selected = read(RUN / 'configs/daso_selected.json')
    scout = read(RUN / 'results/validation_latency_scout.json', selected['latency_scout_sha256'])
    assert scout['complete'] and scout['selection_rules_sha256'] == formal['selection_rules_sha256']
    for source in scout['records']:
        body = read(source['path'], source['sha256'])
        assert body['complete'] and all(row['exact_equal'] for row in body['weight_audit'].values())
    manifest = read(RUN / 'configs/calibration_manifest.json')
    calibration_maps = set()
    for entry in manifest['entries']:
        mapping = read(RUN / f"configs/policies/{entry['policy_id']}.json", entry['policy_sha256'])['method_map']
        calibration_maps.add(hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest())
        closed = read(RUN / f"results/calibration_complete/{entry['policy_id']}.json")
        assert closed['complete']
        identity = read(closed['export_identity'])
        assert identity['policy_sha256'] == entry['policy_sha256']
        audit = read(RUN / f"results/measurements/{Path(closed['export_identity']).parent.name}.json")
        assert audit['complete'] and all(row['exact_equal'] for row in audit['weight_audit'].values())
        for source in closed['quality']:
            result = read(source['path'], source['sha256'])
            assert result['complete'] and len(result['blocks']) == 100
    assert len(calibration_maps) == len(manifest['entries'])
    trace = read(RUN / 'results/hessian_trace_p16_b4_s256.json')
    assert trace['complete'] and len(trace['records']) == 448
    assert trace['args']['probes'] == 16 and trace['args']['blocks'] == 4 and trace['args']['sequence_length'] == 256
    assert len({row['module'] for row in trace['records']}) == 448
    assert all(len(row['trace_samples']) == 64 and all(math.isfinite(value) for value in row['trace_samples']) for row in trace['records'])
    assert read(RUN / 'results/prompt_label_reuse_check.json')['complete']
    read(RUN / 'results/model_identity.json')
    read(RUN / 'configs/canonical_identity.json')
    read(RUN / 'configs/vllm_precompiled_binaries.json')
    read(RUN / 'configs/environment_comparison.json')
    read(RUN / 'configs/engine_frozen.json')
    by_key = {case['case_key']: case for case in formal['unique_cases']}
    test_path = Path(design['quality_samples']['test'])
    assert hashlib.sha256(test_path.read_bytes()).hexdigest() == design['quality_sha256']['test']
    test_tokens = torch.load(test_path, weights_only=True)
    quality, baselines = {}, {}
    for key, case in by_key.items():
        policy = read(RUN / f"configs/policies/{case['policy_id']}.json")
        signature = hashlib.sha256(json.dumps(policy['method_map'], sort_keys=True).encode()).hexdigest()
        assert signature == case['map_signature'] and len(policy['method_map']) == 256
        workload = next(row for row in design['workloads'] if row['scenario'] == case['scenario'])
        path = RUN / f"results/quality/test/{case['policy_id']}/{workload['label']}.json"
        closure = read(RUN / f"results/quality_closure/test/{case['policy_id']}/{workload['label']}.json")
        assert closure['complete']
        result = read(path, closure['result_sha256'])
        audit = read(closure['measurement'], closure['measurement_sha256'])
        assert audit['complete'] and all(row['exact_equal'] for row in audit['weight_audit'].values())
        assert result['complete'] and len(result['blocks']) == 100
        assert result['sample_sha256'] == design['quality_sha256']['test']
        assert result['policy_sha256'] == hashlib.sha256((RUN / f"configs/policies/{case['policy_id']}.json").read_bytes()).hexdigest()
        assert [row['block'] for row in result['blocks']] == list(range(100))
        assert all(row['modules_scheduled'] == 256 for row in result['phase_checks'])
        assert [row['batch'] for row in result['phase_checks']] == [8] * 12 + [4]
        if case['scenario'] == 'prefill_decode':
            captures = RUN / f"results/captures/test/{case['policy_id']}/{workload['label']}/{Path(closure['measurement']).stem}"
            for block in result['blocks']:
                path = captures / f"{block['block']:04d}.jsonl"
                SOURCES[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
                tokens = [json.loads(line) for line in path.read_text().splitlines()]
                assert [row['step'] for row in tokens] == list(range(64))
                assert [row['target_id'] for row in tokens] == test_tokens[block['block'], 2048:2112].tolist()
                assert abs(-sum(row['logprob'] for row in tokens) / 64 - block['avg_nll']) < 1e-12
        quality[key] = result
        if all(method == 'dense_bf16' for pair in policy['method_map'].values() for method in pair.values()):
            baselines[case['scenario']] = key
    speeds, memories = {key: [] for key in by_key}, {key: [] for key in by_key}
    nvml_idle = read(RUN / 'configs/nvml_baseline.json')['gpus'][0]['idle_allocated_bytes']
    for round_index in range(3):
        round_record = read(RUN / f'results/speed_rounds/round{round_index}.json')
        assert round_record['complete'] and round_record['round'] == round_index
        assert round_record['formal_points_sha256'] == hashlib.sha256((RUN / 'configs/formal_points.json').read_bytes()).hexdigest()
        measurements = []
        for source in round_record['records']:
            body = read(source['path'], source['sha256'])
            assert body['complete'] and body['mode'] == 'speed' and body['gpu'] == '0'
            assert body['engine'] == design['engine'] and body['placement']['cpu_affinity'] == list(range(8))
            assert all(row['exact_equal'] for row in body['weight_audit'].values())
            for case in body['cases']:
                assert len(case['samples']) == 15
                timed = [row for row in case['samples'] if not row['warmup']]
                assert len(timed) == 10
                for metric, value in case['medians'].items():
                    assert stats.median(row[metric] for row in timed) == value
                assert all(len(row['output_token_ids']) == 8 and all(len(ids) == case['case']['output']
                    for ids in row['output_token_ids']) for row in case['samples'])
            telemetry = Path(source['path']).stem
            samples = [json.loads(line) for line in (RUN / f'logs/{telemetry}.resources.jsonl').read_text().splitlines()]
            assert samples
            assert all(item['pid'] in row['owned_pids'] for row in samples for gpu in row['gpus'] for item in gpu['processes'])
            body['sampled_nvml_peak_additional_bytes'] = max(row['gpus'][0]['allocated_bytes'] for row in samples) - nvml_idle
            measurements.append(body)
        first, last = measurements[0], measurements[-1]
        assert first['policy'] == last['policy'] == 'uniform_dense_bf16'
        for key, case in by_key.items():
            scenario = case['scenario']
            start = next(row['medians'] for row in first['cases'] if row['case']['scenario'] == scenario)
            end = next(row['medians'] for row in last['cases'] if row['case']['scenario'] == scenario)
            reference = {metric: stats.mean([start[metric], end[metric]]) for metric in start}
            body = first if key == baselines[scenario] else next(body for body in measurements[1:-1]
                if body['policy'] == case['policy_id'] and any(row['case']['scenario'] == scenario for row in body['cases']))
            medians = reference if key == baselines[scenario] else next(row['medians'] for row in body['cases'] if row['case']['scenario'] == scenario)
            speeds[key].append({'case': medians, 'reference': reference,
                'bf16_drift': abs(end['e2e_ms'] - start['e2e_ms']) / reference['e2e_ms']})
            memories[key].append({field: body[field] for field in ['peak_allocated_bytes', 'peak_reserved_bytes',
                'peak_host_rss_bytes', 'host_weight_cache_bytes', 'sampled_nvml_peak_additional_bytes']})
    tasks = {key: {} for key in by_key}
    answers = {}
    for job in formal['downstream_jobs']:
        closed = read(RUN / f"results/downstream_complete/{job['id']}.json")
        assert closed['complete'] and closed['job'] == job
        result = read(closed['result'], closed['result_sha256'])
        assert result['complete'] and result['policy'] == job['policy_id'] and result['task'] == job['task']
        if job['kind'] == 'multiple_choice':
            tasks[job['case_key']][job['task']] = result['metric_value']
            expected = {'wikitext': 62, 'arc_easy': 2376, 'arc_challenge': 1172, 'winogrande': 1267, 'mmlu': 14042}
            assert result['samples'] == expected[job['task']] and result['limit'] is None and result['num_fewshot'] == 0
        else:
            assert result['samples'] == job['end'] - job['begin'] and result['max_new_tokens'] == 256 and result['normal_eos']
            source = Path(result['answers'])
            SOURCES[str(source.relative_to(REPO))] = hashlib.sha256(source.read_bytes()).hexdigest()
            rows = [json.loads(line) for line in source.read_text().splitlines()]
            assert [row['question_index'] for row in rows] == list(range(job['begin'], job['end']))
            assert all(len(row['generated_token_ids']) <= 256 for row in rows)
            answers.setdefault((job['case_key'], job['task']), []).extend(rows)
    for key, case in by_key.items():
        if case['scenario'] == 'prefill_decode':
            for task, count in [('cnn_dm_1000', 1000), ('dsum', 1500), ('IWSLT', 333)]:
                result = read(RUN / f"results/task_metrics/{case['policy_id']}/{task}.json")
                assert result['complete'] and result['samples'] == count
                actual = sorted(answers[key, task], key=lambda row: row['question_index'])
                expected = [json.loads(line) for line in (RUN / f'data/generation_{task}.jsonl').read_text().splitlines()]
                assert [row['question_index'] for row in actual] == list(range(count))
                assert [(row['question_id'], row['reference']) for row in actual] == [(row['question_id'], row['reference']) for row in expected]
                tasks[key][task] = result['sacre_bleu'] if task == 'IWSLT' else result['rougeL_percent']
    rows = []
    for point in formal['points']:
        key, scenario = point['case_key'], point['scenario']
        delta, se = paired(quality[key]['blocks'], quality[baselines[scenario]]['blocks'])
        mapping = read(RUN / f"configs/policies/{point['policy_id']}.json")['method_map']
        assert hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest() == by_key[key]['map_signature']
        speed = summarize_speed(speeds[key])
        task = tasks[key]
        average = (100 * stats.mean(task[name] for name in ['arc_easy', 'arc_challenge', 'winogrande', 'mmlu'])
                   if scenario == 'prefill_only' else stats.mean(task[name] for name in ['cnn_dm_1000', 'dsum', 'IWSLT']))
        row = {'family': point['family'], 'scenario': scenario,
            'budget': point.get('quality_budget', point.get('paired_daso_budget')), 'policy_id': point['policy_id'],
            'case_key': key, 'nll': stats.mean(item['avg_nll'] for item in quality[key]['blocks']),
            'delta_nll': delta, 'paired_se': se, 'task_average': average, **task,
            'torch_peak_allocated_gib': stats.median(item['peak_allocated_bytes'] for item in memories[key]) / 1024**3,
            'torch_peak_reserved_gib': stats.median(item['peak_reserved_bytes'] for item in memories[key]) / 1024**3,
            'sampled_nvml_peak_additional_gib': stats.median(item['sampled_nvml_peak_additional_bytes'] for item in memories[key]) / 1024**3,
            'host_phase_cache_gib': stats.median(sum(item['host_weight_cache_bytes'].values()) for item in memories[key]) / 1024**3,
            'peak_process_rss_gib': stats.median(item['peak_host_rss_bytes'] for item in memories[key]) / 1024**3,
            'variation_flag': speed['variation_flag'], 'hawq_cap_relaxed': point.get('relaxed_infeasible', False)}
        row['requested_linear_limit_ms'] = point.get('requested_linear_limit_ms')
        row['actual_linear_limit_ms'] = point.get('linear_limit_ms')
        for metric, values in speed.items():
            if isinstance(values, dict):
                row[metric] = values['milliseconds']; row[metric + '_speedup'] = values['speedup']
        rows.append(row)
        write_json(RUN / f'results/final_detail/{key}.json', {'speed': speed, 'memory': memories[key],
            'method_composition': {phase: dict(Counter(pair[phase + '_method'] for pair in mapping.values())) for phase in ['prefill', 'decode']}})
    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with (OUT / 'results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    exploration = []
    for row in selected['all_validation_candidates']:
        measured = scout['points'].get(row['scenario'] + '_' + row['map_signature'])
        exploration.append({**row, 'exploratory_e2e_ms': measured['medians']['e2e_ms'] if measured else '',
            'exploratory_cycle_ms': measured['medians']['cycle_ms'] if measured else ''})
    with (OUT / 'validation_search.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exploration[0])); writer.writeheader(); writer.writerows(exploration)
    preparation = {'calibration_policies': len(manifest['entries']), 'windows_per_scenario': 100,
        'canonical_pruning_seconds': {name: read(RUN / f'results/canonical_{name}.json')['elapsed_seconds']
            for name in ['sparse_bf16', 'sparse_nvfp4']}, 'hessian_seconds': trace['elapsed_seconds'],
        'linear_profile_seconds': read(RUN / 'results/profile_complete.json')['seconds'],
        'calibration_metrics': {scenario: read(RUN / f'results/quality_model/{scenario}/metrics.json')
            for scenario in ['prefill_only', 'prefill_decode']},
        'solver_details': {scenario: read(RUN / f'results/quality_model/{scenario}/proposals.json')
            for scenario in ['prefill_only', 'prefill_decode']},
        'proxy_budget_backoff': refinement,
        'scope': 'Pruning and Hessian compute times exclude checkpoint serialization. Calibration labels, dependency setup, '
                 'data preparation and policy validation are additional work; solve times assume all profiles/features/calibration exist.'}
    first_command = min(read(path)['time_ns'] for path in (RUN / 'logs').glob('calibration_*.command.json'))
    last_closure = max((RUN / f"results/calibration_complete/{entry['policy_id']}.json").stat().st_mtime_ns
                       for entry in manifest['entries'])
    preparation['calibration_observed_wall_interval_seconds'] = (last_closure - first_command) / 1e9
    preparation['calibration_interval_definition'] = 'Earliest calibration command timestamp to final completion-marker write; includes failed attempts, export, hashing, loading, compilation and waiting. Not summed GPU compute time.'
    write_json(OUT / 'preparation.json', preparation)
    write_json(OUT / 'results.json', {'points': rows, 'formal_points': formal, 'source_files': SOURCES})
    lines = ['# Qwen3-32B · RTX PRO 6000 完整对照', '',
        '完整64层、32,762,123,264参数；BF16、DASO及HAWQ-V3适配共用本轮phase运行时。正式速度为单卡GPU0，CPU0–7、NUMA0；B8/S2048，Prefill输出1，P+D输出64，固定8GiB BF16 KV。', '',
        'DASO先搜索六档预算及四组阶段组合，再根据紧预算验证反馈追加0.003/0.005两档搜索，保持质量模型不变。以mean+2SE+0.005余量检查质量，对合格候选映射逐一探索计时，按E2E选出预算0.02/0.07代表点，再独立测正式三轮。HAWQ请求匹配其Linear代理延迟上限，预算标签不表示HAWQ满足同一真实NLL约束。所有策略在test和下游测量前冻结。', '']
    for scenario, title in [('prefill_only', 'Prefill-only'), ('prefill_decode', 'Prefill+Decode')]:
        lines += [f'## {title}', '', '| 方法 | 预算标签 | Test ΔNLL | Avg ↑ | TTFT× | E2E× | cycle× | Torch峰值 GiB | Host阶段缓存 GiB |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for row in [row for row in rows if row['scenario'] == scenario]:
            flag = ('†' if row['variation_flag'] else '') + ('‡' if row['hawq_cap_relaxed'] else '')
            label = '—' if row['budget'] is None else f"{row['budget']:g}"
            lines.append(f"| {row['family']} | {label} | {row['delta_nll']:.6f} | {row['task_average']:.2f} | {row['ttft_ms_speedup']:.3f} | {row['e2e_ms_speedup']:.3f}{flag} | {row['cycle_ms_speedup']:.3f} | {row['torch_peak_allocated_gib']:.2f} | {row['host_phase_cache_gib']:.2f} |")
        lines.append('')
    if any(row['hawq_cap_relaxed'] for row in rows):
        lines += ['‡该HAWQ点无法在其候选空间中达到请求的DASO代理延迟上限，已按预设规则放宽至HAWQ最小可行延迟；这些点不是相同代理上限内的对照。请求/实际上限保留在CSV和冻结配置中。', '']
    lines += ['## 口径与原始证据', '',
        '- 每点100个固定WikiText test窗口，ΔNLL单位nats/token。逐窗配对SE保留在CSV；共享/可能重叠文本窗口，不把SE解释为独立样本置信保证。',
        '- Prefill Avg为ARC-E、ARC-C、Wino、MMLU百分制等权均值；P+D Avg为CNN/DM、DialogSum ROUGE-L%与IWSLT corpus BLEU等权均值，二者不可互比。WikiText word perplexity和全部任务分项见CSV。生成上限256、正常EOS，不是测速固定64。',
        '- 三轮，每点5次warmup、10次计时，先取轮内中位数，再用本轮BF16首尾中位数的算术均值归一化，报告三轮比值算术均值。†表示E2E轮间范围/均值或任一BF16首尾漂移超过5%；完整轮次均保留。TPOT、绝对毫秒、恢复和逐轮比值见CSV/详细JSON。',
        '- 显存为Torch分配峰值，单列持久Host阶段缓存；进程RSS包括加载，内存盘导出不等于持久阶段缓存。未将这些不同资源相加为显存。',
        '- HAWQ使用本模型P16/B4/S256 signed Hessian迹及FP4/INT8/BF16 PTQ后端；不是原CNN QAT/TVM逐项复现。DASO有八种候选，因此不作为仅分配器变化的因果消融。',
        '- 相同策略映射共享测量证据，不是独立重复。256模块组成在详细JSON中保留；两阶段同一uniform方法的点不称为异构收益，仅阶段格式不同的组合不称为逐层异构收益。',
        '- 旧机器与本轮二进制环境有显式差异；新32B校准、新PRO6000画像全部重新测量。不得把既有准备后的求解耗时解释为完整准备耗时。', '']
    for target, label in [(OUT / 'results.csv', '完整分项CSV'), (OUT / 'results.json', '汇总及逐文件SHA'),
        (OUT / 'validation_search.csv', '全部validation搜索候选与探索计时（非正式test结果）'),
        (RUN / 'results/validation_latency_scout.json', '独立探索计时与全部尝试入口'),
        (RUN / 'configs/formal_points.json', '冻结策略与别名'), (RUN / 'configs/selection_rules.json', '预先冻结的选择/聚合规则'),
        (RUN / 'results/speed_rounds', '三轮原始测量入口'), (RUN / 'results/final_detail', '逐轮速度与模块组成'),
        (RUN / 'configs/environment_comparison.json', '环境差异'), (RUN / 'results/hessian_trace_p16_b4_s256.json', 'HAWQ完整迹')]:
        lines.append(f'- [{label}]({os.path.relpath(target, OUT)})')
    lines += ['', '## 准备成本', '',
        f"两套canonical剪枝分别{preparation['canonical_pruning_seconds']['sparse_bf16']:.1f}/{preparation['canonical_pruning_seconds']['sparse_nvfp4']:.1f}秒，HAWQ主迹{preparation['hessian_seconds']:.1f}秒，Linear画像{preparation['linear_profile_seconds']:.2f}秒；剪枝时间不含权重序列化。72套策略双场景100窗校准，从首次命令到最后闭合标记的观察墙钟区间为{preparation['calibration_observed_wall_interval_seconds'] / 3600:.2f}小时，包含失败尝试、导出、hash、加载、编译和等待，不等于GPU计算时间。另需局部特征、依赖和validation闭环，不能将已有这些资料后的求解耗时作为全部准备耗时。", '',
        '[拟合留出误差、求解耗时和准备阶段分项](preparation.json)。']
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    write_json(RUN / 'results/final_audit_complete.json', {'complete': True, 'display_points': len(rows),
        'unique_cases': len(by_key), 'calibration_policies': len(manifest['entries']), 'speed_rounds': 3,
        'unique_calibration_maps': len(calibration_maps),
        'task_jobs': len(formal['downstream_jobs']), 'source_files': SOURCES, 'report': str(OUT / 'README.md')})
    from update_reply import main as update_reply
    update_reply()


if __name__ == '__main__':
    main()
