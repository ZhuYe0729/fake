"""Verify complete raw evidence and seal the isolated table-completion run."""
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
REPO = RUN.parents[3]
TABLE = REPO / 'artifacts/rebuttal/responses/reviewer2/qwen3_table2'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert read(RUN / 'results/worker.json')['status'] == 'complete'
    design = read(RUN / 'configs/design.json')
    complete = read(RUN / 'results/table_complete.json')
    assert complete['missing_cells'] == 0
    rows = list(csv.DictReader((TABLE / 'main_table.csv').open()))
    assert len(rows) == 17 and all(len(row) == 13 for row in rows)
    assert all(math.isfinite(float(v)) for row in rows for k, v in row.items() if k not in ['model', 'method'])
    assert '待补' not in (TABLE / 'README.md').read_text()
    for source, sha in read(TABLE / 'provenance.json')['sources_sha256'].items():
        assert digest(REPO / source) == sha, source
    for rid, sha in read(RUN / 'reference/protected_manifests.json').items():
        assert digest(RUN.parent / rid / 'results/manifest.json') == sha
    commands = list((RUN / 'logs').glob('*.command.json'))
    for path in commands:
        record = read(path)
        result = path.with_name(path.name.replace('.command.json', '.completion.json'))
        assert result.exists() and read(result)['returncode'] == 0, path
        for name, sha in record['code_hashes'].items():
            assert digest(RUN / 'code_versions' / record['code_version'] / name) == sha
    processes = cases = samples = 0
    for spec in design['rounds']:
        for pid in [*spec['order'], 'uniform_dense_bf16_bracket_end']:
            data = read(RUN / f'results/speed/node0/round{spec["round"]}/{pid}.json')
            assert data['input_sha256'] == design['speed_inputs_sha256']
            assert data['first_token_observer'] == 'engine_core_before_FINAL_ONLY_filter'
            processes += 1
            for case in data['cases']:
                cases += 1
                timed = [s for s in case['samples'] if not s['warmup']]
                assert len(timed) == 8
                samples += len(timed)
                if case['output'] == 64:
                    assert all(0 < s['ttft_observed_ms'] < s['e2e_ms'] * .9 for s in timed)
    assert (processes, cases, samples) == (69, 102, 816)
    for job in design['quality_jobs']:
        if job['kind'] == 'tasks':
            paths = list((RUN / f'results/tasks/{job["policy"]}').glob('*/full/result.json'))
            assert len(paths) == 5 and all(read(p)['limit'] is None for p in paths)
        else:
            for task, count in [('cnn_dm_1000', 1000), ('dsum', 1500), ('IWSLT', 333)]:
                base = RUN / f'results/generation/{job["policy"]}/{task}'
                assert read(base / 'metrics_0000_end.json')['samples'] == count
                answers = [json.loads(line) for line in (base / 'answers_0000_end.jsonl').read_text().splitlines()]
                expected = [json.loads(line) for line in (RUN / f'data/generation_{task}.jsonl').read_text().splitlines()]
                assert len(answers) == count
                assert [a['question_id'] for a in answers] == [a['question_id'] for a in expected]
    target = RUN / 'results/table'
    target.mkdir(exist_ok=True)
    for p in TABLE.iterdir():
        if p.is_file():
            shutil.copy2(p, target / p.name)
    files = {}
    for p in sorted(RUN.rglob('*')):
        relative = str(p.relative_to(RUN))
        if relative in ['results/manifest.json', 'results/final_audit.json']:
            continue
        if p.is_symlink():
            files[relative] = {'symlink': str(p.resolve())}
            if p.is_file():
                files[relative]['sha256'] = digest(p)
        elif p.is_file():
            files[relative] = {'sha256': digest(p), 'bytes': p.stat().st_size}
    manifest = RUN / 'results/manifest.json'
    manifest.write_text(json.dumps({'files': files}, indent=2) + '\n')
    audit = {**complete, 'verified': True, 'commands': len(commands), 'speed_processes': processes,
        'speed_cases': cases, 'timed_generations': samples, 'protected_runs_unchanged': 12,
        'archived_files': len(files), 'manifest_sha256': digest(manifest),
        'scope': 'All retained main-table rows; no new policy selection; all code/logs/answers preserved.'}
    (RUN / 'results/final_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
