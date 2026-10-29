"""Copy completed evidence into the five human-reviewer response directories."""
import hashlib
import json
import shutil
from pathlib import Path

from common import RUN, REPO, write_json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    responses = REPO / 'artifacts/rebuttal/responses'
    groups = {
        1: ('质量预算、实测约束与阶段复用内存', ['selected_test_constraints.csv', 'raw_proxy_test_constraints.csv',
            'constraints_summary.json', 'runtime_overhead.csv', 'overhead.md'], ['analyze_constraints.py', 'analyze_overhead.py']),
        2: ('Qwen3完整主表与强化HAWQ对照', ['hawq_latency_controls.csv', 'hawq_controls_summary.json',
            'heldout_frontier_points.csv', 'heldout_frontier.pdf', 'fixed_main_table.csv'], ['analyze_hawq.py', 'plot_frontiers.py', 'build_report.py']),
        3: ('Qwen3泛化、代表负载与建模开销', ['representative_workloads.csv', 'representative_workloads.md',
            'preparation_summary.csv', 'overhead.md', 'graph_action_profile.csv'], ['analyze_workloads.py', 'analyze_overhead.py']),
        4: ('负载配置、硬件画像与搜索成本', ['representative_workloads.csv', 'representative_workloads.md',
            'graph_action_profile.csv', 'preparation_summary.csv', 'overhead.md'], ['analyze_workloads.py', 'analyze_overhead.py']),
        5: ('反向约束、质量代理与实际违约', ['selected_test_constraints.csv', 'raw_proxy_test_constraints.csv',
            'constraints_summary.json', 'heldout_frontier_points.csv', 'heldout_frontier.pdf'], ['analyze_constraints.py', 'plot_frontiers.py'])}
    packages = {}
    for number, (topic, names, scripts) in groups.items():
        root = responses / f'reviewer{number}/graph_run015'
        for directory in ['results', 'code', 'configs']:
            (root / directory).mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy2(RUN / 'results' / name, root / 'results' / name)
        for name in ['selected_reanalysis.json', 'selection_rules.json', 'representative_workloads.json',
                     'hawq_latency_controls.json', 'hawq_robustness.json']:
            shutil.copy2(RUN / 'configs' / name, root / 'configs' / name)
        # Entry point runs the preserved source tree; copied configs/results are
        # evidence snapshots, not a relocated runtime or an independent package.
        wrapper = '''"""Rebuild analysis using the preserved run015 source tree and raw evidence."""
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[6]
SCRIPTS = %r
for script in SCRIPTS:
    subprocess.run([sys.executable, str(ROOT / 'artifacts/rebuttal/runs/015/code' / script)], check=True)
''' % scripts
        (root / 'code/reproduce_analysis.py').write_text(wrapper)
        (root / 'README.md').write_text(f'# Run015：{topic}\n\n'
            '本目录保存对应人工reviewer的最终分析结果及冻结配置快照。'
            '完整代码、原始数据/日志和失败记录保留在[run015](../../../runs/015/README.md)，'
            '源代码没有搬迁。`code/reproduce_analysis.py`在同一仓库中调用run015源程序重建分析；需要原实验Python环境和原始归档。\n\n'
            '[完整17行主表](../../reviewer2/qwen3_table2/README.md)使用统一RTX5090 GPU0、NUMA0、'
            'B8/S2048/O1或64、6GiB BF16 KV、decode Graph、三轮独立进程和BF16首尾锚点。'
            '新配置补完整质量，旧相同有效map按数据/权重/协议核验后继承。\n\n'
            '代表负载为固定主配置迁移，不能当作逐负载重优化或逐负载质量保证。'
            'validation/test split不同，但test窗口沿用历史冻结集合；不宣称跨所有轮次全新未见测试集。'
            'epsilon是NLL代理预算，HAWQ是Hessian/ILP的PTQ移植。实测E2E、恢复完整cycle、准备和初始化分别报告；不以代理最优代替真实最优。\n')
        sources = {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file() and p.name != 'SOURCE.json'}
        write_json(root / 'SOURCE.json', {'run': '015', 'topic': topic, 'files_sha256': sources,
            'analysis_source_sha256': {name: sha(RUN / 'code' / name) for name in scripts},
            'selection_sha256': sha(RUN / 'configs/selected_reanalysis.json'),
            'scope': 'Evidence snapshots and a repository-relative analysis entry point; full executable source and raw records remain in run015.'})
        assert all(sha(root / name) == digest for name, digest in sources.items())
        packages[number] = {'directory': str(root.relative_to(REPO)), 'files': len(sources),
                            'source_index_sha256': sha(root / 'SOURCE.json')}
    write_json(RUN / 'results/response_packages.json', packages)
    index = responses / 'README.md'
    if '## 统一Graph运行时与重分析（run015）' not in index.read_text():
        with index.open('a') as handle:
            handle.write('\n## 统一Graph运行时与重分析（run015）\n\n'
                'W4A16 TPOT诊断后，统一执行路径、新画像和重新分配后的比较见'
                '[完整主表](reviewer2/qwen3_table2/README.md)。旧eager结果作为对应运行时的历史证据保留，不能与新Graph数字混排。\n\n'
                + ' · '.join(f'[reviewer{i}最终补充](reviewer{i}/graph_run015/README.md)' for i in groups) + '\n')


if __name__ == '__main__':
    main()
