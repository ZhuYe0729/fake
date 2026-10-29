"""Complete the main three campaigns first, then separately anchored supplements."""
import sys
import time

from launch import RUN, checkpoint, cleanup, invoke, read, write


def main():
    assert read(RUN / 'results/full_quality_complete.json')['complete']
    assert read(RUN / 'results/hawq_robustness_quality_complete.json')['complete']
    design = read(RUN / 'configs/design.json')
    from telemetry import nvml
    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    for spec in design['rounds']:
        group = spec.get('group', 'main')
        order = [*spec['order'], 'uniform_dense_bf16']
        for index, pid in enumerate(order):
            end = index == len(order) - 1
            name = pid + ('_extra' if group == 'supplement' else '') + ('_bracket_end' if end else '')
            target = RUN / f'results/speed/node0/round{spec["round"]}/{name}.json'
            if target.exists():
                continue
            while nvml.nvmlDeviceGetMemoryInfo(handle).used > 512 * 1024**2:
                time.sleep(10)
            model = checkpoint(pid, 0)
            write(RUN / 'results/formal_progress.json', {'status': 'running', 'group': group,
                  'round': spec['round'], 'index': index, 'total_policies': len(order), 'policy': pid})
            invoke([sys.executable, RUN / 'code/formal_benchmark.py', '--checkpoint', model,
                    '--policy', pid, '--round', str(spec['round']), '--record-id', name,
                    '--outputs', *map(str, design['policies'][pid]['outputs'])],
                   f'formal_round{spec["round"]}_{name}', 0)
            cleanup(model, f'formal_round{spec["round"]}_{name}')
            print('formal complete', group, spec['round'], name, flush=True)
        if group == 'main' and spec['round'] == 2:
            marker = RUN / 'results/priority_main_complete.json'
            if not marker.exists():
                write(marker, {'complete': True, 'scope': 'All three main campaigns with original repetitions; supplements remain.'})
            published = RUN / 'results/priority_main_published.json'
            if not published.exists():
                invoke([sys.executable, RUN / 'code/build_report.py', '--main-only'], 'priority_main_report', 0)
                write(published, {'complete': True})
    write(RUN / 'results/formal_progress.json', {'status': 'complete', 'group': 'all'})


if __name__ == '__main__':
    main()
