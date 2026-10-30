"""Publish the usable core table before the longer downstream jobs finish."""
import csv
import statistics

from common import RUN, write_json
from final_report import OUT, paired, read, summarize_speed


def main():
    assert read(RUN / 'results/test_complete.json')['complete']
    assert read(RUN / 'results/formal_speed_complete.json')['complete']
    formal = read(RUN / 'configs/formal_points.json')
    design = read(RUN / 'configs/design.json')
    by_key = {row['case_key']: row for row in formal['unique_cases']}
    baselines = {}
    for row in formal['points']:
        if row['family'] == 'BF16':
            baselines[row['scenario']] = row['case_key']
    measurements = []
    for index in range(3):
        record = read(RUN / f'results/speed_rounds/round{index}.json')
        assert record['complete']
        measurements.append([read(source['path'], source['sha256']) for source in record['records']])
    rows = []
    for point in formal['points']:
        case, scene = by_key[point['case_key']], point['scenario']
        rounds, allocated, host = [], [], []
        for bodies in measurements:
            start = next(row['medians'] for row in bodies[0]['cases'] if row['case']['scenario'] == scene)
            end = next(row['medians'] for row in bodies[-1]['cases'] if row['case']['scenario'] == scene)
            reference = {metric: statistics.mean([value, end[metric]]) for metric, value in start.items()}
            body = bodies[0] if point['case_key'] == baselines[scene] else next(body for body in bodies[1:-1]
                if body['policy'] == case['policy_id'] and any(row['case']['scenario'] == scene for row in body['cases']))
            values = reference if point['case_key'] == baselines[scene] else next(row['medians'] for row in body['cases'] if row['case']['scenario'] == scene)
            rounds.append({'case': values, 'reference': reference, 'bf16_drift': abs(end['e2e_ms'] - start['e2e_ms']) / reference['e2e_ms']})
            allocated.append(body['peak_allocated_bytes']); host.append(sum(body['host_weight_cache_bytes'].values()))
        workload = next(row['label'] for row in design['workloads'] if row['scenario'] == scene)
        quality = read(RUN / f"results/quality/test/{case['policy_id']}/{workload}.json")
        baseline = read(RUN / f"results/quality/test/{by_key[baselines[scene]]['policy_id']}/{workload}.json")
        delta, se = paired(quality['blocks'], baseline['blocks'])
        speed = summarize_speed(rounds)
        row = {'family': point['family'], 'scenario': scene, 'budget': point.get('quality_budget', point.get('paired_daso_budget')),
            'policy_id': point['policy_id'], 'test_delta_nll': delta, 'paired_se': se,
            'torch_peak_allocated_gib': statistics.median(allocated) / 1024**3,
            'persistent_host_phase_cache_gib': statistics.median(host) / 1024**3, 'variation_flag': speed['variation_flag'],
            'hawq_cap_relaxed': point.get('relaxed_infeasible', False),
            'requested_linear_limit_ms': point.get('requested_linear_limit_ms'), 'actual_linear_limit_ms': point.get('linear_limit_ms')}
        for metric, value in speed.items():
            if isinstance(value, dict):
                row[metric] = value['milliseconds']; row[metric + '_speedup'] = value['speedup']
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'core_results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader(); writer.writerows(rows)
    lines = ['# 完整32B固定负载核心结果', '',
        '本表仅汇总冻结策略后的100窗test NLL、三轮正式速度及内存；ΔNLL单位nats/token，原任务精度和完整终审见最终报告。B8/S2048，Prefill输出1，P+D输出64，8GiB BF16 KV。', '',
        '| 方法 | 场景 | 预算标签 | Test ΔNLL | TTFT× | E2E× | cycle× | Torch分配峰值 GiB | Host阶段缓存 GiB |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        budget = '—' if row['budget'] is None else f"{row['budget']:g}"
        flag = ('†' if row['variation_flag'] else '') + ('‡' if row['hawq_cap_relaxed'] else '')
        lines.append(f"| {row['family']} | {row['scenario']} | {budget} | {row['test_delta_nll']:.6f} | {row['ttft_ms_speedup']:.3f} | {row['e2e_ms_speedup']:.3f}{flag} | {row['cycle_ms_speedup']:.3f} | {row['torch_peak_allocated_gib']:.2f} | {row['persistent_host_phase_cache_gib']:.2f} |")
    lines += ['', '†保留超过5%的轮间E2E范围或BF16首尾漂移。三轮比值均值；每轮先对10次计时取中位数，再以本轮BF16首尾中位数均值归一化。HAWQ匹配Linear代理延迟上限，预算标签不表示相同真实质量约束；相同映射共享证据。', '',
        '[完整毫秒、TPOT及恢复分项](core_results.csv)。']
    if any(row['hawq_cap_relaxed'] for row in rows):
        lines += ['', '‡该HAWQ点的请求代理延迟不可达，已按预设规则放宽到其候选空间的最小可行值；不是相同代理上限内的对照，请求/实际上限见CSV。']
    (OUT / 'CORE_RESULTS.md').write_text('\n'.join(lines) + '\n')
    write_json(RUN / 'results/core_complete.json', {'complete': True, 'points': len(rows),
        'report': str(OUT / 'CORE_RESULTS.md'), 'scope': 'Test NLL and three formal timing rounds only; full downstream and final audit are separate.'})


if __name__ == '__main__':
    main()
