"""Verify completed evidence, preserve prior runs, and seal a reproducible package."""
import hashlib
import json
from pathlib import Path

from common import RUN, REPO, write_json


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    table = REPO / 'artifacts/rebuttal/responses/reviewer2/qwen3_table2'
    assert read(RUN / 'results/table_complete.json')['complete']
    assert read(RUN / 'results/formal_progress.json')['status'] == 'complete'
    assert read(RUN / 'results/full_quality_complete.json')['complete']
    assert read(RUN / 'results/workload_progress.json')['status'] == 'complete'
    assert read(RUN / 'results/workload_smoke_gate.json')['passed']
    assert read(RUN / 'results/graph_bindings_numerical_gate.json')['passed']
    assert read(RUN / 'results/runtime_gate_device_alpha.json')['passed']
    design = read(RUN / 'configs/design.json')
    assert sha(RUN / 'configs/selected_reanalysis.json') == design['selection_sha256']
    selection = read(RUN / 'configs/selected_reanalysis.json')
    assert sha(RUN / 'configs/reanalysis_pool.json') == selection['pool_sha256']
    assert sha(RUN / 'configs/selection_rules.json') == selection['rules_sha256']
    assert read(RUN / 'configs/test_requests.json')['selection_sha256'] == design['selection_sha256']
    quality_results = 0
    for split in ['validation', 'test']:
        data_sha = sha(RUN / f'data/{split}_64_2112.pt')
        for path in (RUN / f'results/{split}').glob('*/*.json'):
            value = read(path)
            assert value['data_sha256'] == data_sha and value['quality_score_batch'] == 8, str(path)
            assert len(value['blocks']) == 64, str(path)
            expected = 2047 if path.stem == 'prefill_only' else 64
            assert [b['block'] for b in value['blocks']] == list(range(64)), str(path)
            assert all(b['token_count'] == expected for b in value['blocks']), str(path)
            assert value['token_count'] == expected * 64, str(path)
            quality_results += 1
    protected_files = 0
    for rid, digest in read(RUN / 'reference/protected_manifests.json').items():
        base = RUN.parent / rid
        manifest = base / 'results/manifest.json'
        assert sha(manifest) == digest, str(manifest)
        entries = read(manifest)['files']
        entries = entries.items() if isinstance(entries, dict) else ((e['path'], e) for e in entries)
        for name, entry in entries:
            path = base / name
            if 'symlink' in entry:
                assert path.is_symlink() and str(path.resolve()) == entry['symlink'], str(path)
            if 'bytes' in entry:
                assert path.stat().st_size == entry['bytes'], str(path)
            if 'sha256' in entry:
                assert sha(path) == entry['sha256'], str(path)
            protected_files += 1
    import common
    binary = common.VLLM / 'vllm/vllm/_C.abi3.so'
    assert sha(binary) == 'a6a469b427f36a61582e1fa99d8f83cf0fdd230e836e5eae7fe01514cd65fc2c'
    for path in (RUN / 'reference/kernel_sources').iterdir():
        original = common.WRAPPER / 'cutlass_wrapper/csrc' / path.name
        assert sha(path) == sha(original), str(original)
    commands, failures = 0, []
    for path in sorted((RUN / 'logs').glob('*.command.json')):
        done = path.with_name(path.name.replace('.command.json', '.completion.json'))
        if not done.exists():
            # The currently running post-selection/audit drivers finish after sealing.
            continue
        commands += 1
        if read(done)['returncode']:
            failures.append({'command': str(path.relative_to(RUN)), **read(done)})
    write_json(RUN / 'results/retained_failures.json', failures)
    (table / 'build_table.py').write_text('''"""Rebuild the uniform-runtime table from frozen run015 evidence."""
import runpy
from pathlib import Path
ROOT = Path(__file__).resolve().parents[5]
runpy.run_path(str(ROOT / "artifacts/rebuttal/runs/015/code/build_report.py"), run_name="__main__")
''')
    diagnosis = table / 'TPOT_DIAGNOSIS.md'
    note = '\n\n## Run015 后续状态\n\n统一运行时、新画像和重新分配后的完整主表已更新至 [README](README.md)。上文为run014历史诊断；旧eager表保存在run013，不再用于新主表排名。\n'
    if '## Run015 后续状态' not in diagnosis.read_text():
        with diagnosis.open('a') as handle:
            handle.write(note)
    active_logs = {p.with_name(p.name.replace('.command.json', '.log')) for p in
        (RUN / 'logs').glob('*.command.json')
        if not p.with_name(p.name.replace('.command.json', '.completion.json')).exists()}
    files = {}
    for path in sorted(RUN.rglob('*')):
        name = str(path.relative_to(RUN))
        if path.is_dir() or name in ['results/manifest.json', 'results/final_audit.json']:
            continue
        if path in active_logs or path.name.endswith('_driver.log'):
            files[name] = {'mutable_orchestration_log': True}
        elif path.is_symlink():
            files[name] = {'symlink': str(path.resolve())}
        else:
            files[name] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    write_json(RUN / 'results/manifest.json', {'files': files,
        'scope': 'Code, configs, raw outputs, failures and reports. External reproducible model/binary dependencies are recorded separately. Active orchestration logs may append their final exit record after this seal.'})
    write_json(RUN / 'results/final_audit.json', {'complete': True, 'verified': True,
        'protected_runs': 14, 'protected_files_verified': protected_files,
        'formal_policies': len(design['policies']), 'main_rows': len(design['rows']),
        'quality_results_verified': quality_results,
        'commands_with_exit_status': commands, 'retained_failed_commands': len(failures),
        'archived_files': len(files), 'manifest_sha256': sha(RUN / 'results/manifest.json'),
        'response_sha256': {p.name: sha(p) for p in table.iterdir() if p.is_file()}})
    print('Sealed', len(files), 'files; verified prior files', protected_files, flush=True)


if __name__ == '__main__':
    main()
