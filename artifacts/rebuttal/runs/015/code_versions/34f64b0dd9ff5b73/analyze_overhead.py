"""Separate amortized preparation, validation search and steady-state runtime cost."""
import csv
import json
import statistics
from collections import defaultdict

from common import RUN, write_json


def read(path):
    return json.loads(path.read_text())


def main():
    stages = defaultdict(list)
    records = []
    for path in sorted((RUN / 'logs').glob('*.command.json')):
        metadata = read(path)
        done = path.with_name(path.name.replace('.command.json', '.completion.json'))
        if not done.exists():
            continue
        completion = read(done)
        command = metadata['command']
        script = next((x.rsplit('/', 1)[-1] for x in command if x.endswith('.py')), '')
        if script == 'export_model.py':
            stage = 'checkpoint export (repeated per experiment)'
        elif script == 'quality.py':
            stage = 'validation NLL' if 'validation' in command else 'independent test NLL'
        elif script == 'benchmark.py' and any(x.startswith('calibration_') for x in command):
            stage = 'candidate calibration speed'
        elif script == 'profile_actions.py':
            stage = 'action profiling process (includes init)'
        elif script == 'select_reanalysis.py':
            stage = 'validation selection CPU process'
        else:
            continue
        row = {'stage': stage, 'seconds': completion['seconds'], 'returncode': completion['returncode'],
            'gpu': metadata['gpu'], 'source': str(path.relative_to(RUN))}
        records.append(row)
        stages[stage].append(row)
    summary = [{'stage': name, 'successful_processes': sum(x['returncode'] == 0 for x in items),
                'failed_processes': sum(x['returncode'] != 0 for x in items),
                'sum_success_process_seconds': sum(x['seconds'] for x in items if x['returncode'] == 0),
                'sum_failure_process_seconds': sum(x['seconds'] for x in items if x['returncode'] != 0)}
               for name, items in stages.items()]
    for filename, rows in [('preparation_process_costs.csv', records), ('preparation_summary.csv', summary)]:
        with (RUN / 'results' / filename).open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    runtime = list(csv.DictReader((RUN / 'results/runtime_overhead.csv').open()))
    lines = ['# 开销与摊销边界', '',
        '阶段耗时分开报告；并行进程耗时之和不等于实际墙钟总时长。导出包含实验隔离所需的重复构建，不能直接当作每次部署必需成本。', '',
        '| Stage | Successful processes | Sum process seconds | Failed processes |',
        '| --- | ---: | ---: | ---: |']
    lines += [f'| {r["stage"]} | {r["successful_processes"]} | {r["sum_success_process_seconds"]:.2f} | {r["failed_processes"]} |' for r in summary]
    profile = read(RUN / 'results/profile_complete.json')
    lines += ['', f'80项算子画像内部计时为{profile["seconds"]:.2f}s，不包含模型/扩展初始化。', '',
        '训练代理、SparseGPT canonical和HAWQ Hessian本轮复用既有结果；从零成本见run004/results/preparation_costs.json，'
        '例如HAWQ Hessian约1765.64s，不能将本轮缓存复用记为零成本复现。', '',
        '运行时初始化包含Graph捕获，二者不可相加；采样NVML最大使用量不是连续测得的瞬时峰值。'
        'GPU权重存储已按实际storage去重；arena表示两阶段按层复用容量，host缓存另列。', '',
        '| Main P+D method | Init s | Capture s (inside init) | Restore ms | Peak allocated GiB | GPU linear storage GiB | Host cache GiB |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    design = read(RUN / 'configs/design.json')
    for spec in design['rows']:
        items = [r for r in runtime if r['policy'] == spec['prefill_decode']['policy'] and r['scenario'] == 'prefill_decode']
        assert len(items) == 3
        med = lambda key: statistics.median(float(x[key]) for x in items)
        lines.append(f'| {spec["method"]} | {med("initialization_ms")/1000:.2f} | '
            f'{med("capture_model_ms")/1000:.2f} | {med("restore_ms"):.2f} | '
            f'{med("torch_peak_allocated_bytes")/2**30:.2f} | {med("gpu_linear_unique_storage_bytes")/2**30:.2f} | '
            f'{med("host_weight_bytes")/2**30:.2f} |')
    lines += ['', '全部policy和每轮原始开销见runtime_overhead.csv；恢复包含在cycle中，主表E2E沿用原协议不含请求前恢复。', '']
    (RUN / 'results/overhead.md').write_text('\n'.join(lines))
    write_json(RUN / 'results/overhead_summary.json', {'stages': summary, 'profile_internal_seconds': profile['seconds'],
        'inherited_preparation_source': '004/results/preparation_costs.json',
        'scope': 'Separates cached preparation, expanded validation search and per-request runtime. No unsupported total-wall-time claim.'})


if __name__ == '__main__':
    main()
