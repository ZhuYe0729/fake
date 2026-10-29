"""Delete only completed, reproducible Llama export weights; retain evidence."""
import argparse
import fcntl
import hashlib
import json
import os
import stat
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROOT = REPO / 'artifacts/rebuttal'
NIGHT = ROOT / 'night151'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path):
    s = path.lstat()
    return {'bytes': s.st_size, 'allocated_bytes': s.st_blocks * 512,
            'inode': s.st_ino, 'device': s.st_dev, 'mtime_ns': s.st_mtime_ns, 'links': s.st_nlink}


def dependencies(rid, policy):
    spec = read(ROOT / f'runs/{rid}/configs/model.json')
    source = Path(spec['model_path'])
    assert (source / 'config.json').is_file() and list(source.glob('*.safetensors')), source
    paths = [source / 'config.json', *source.glob('*.safetensors')]
    methods = {v[k] for v in read(policy)['method_map'].values() for k in ['prefill_method', 'decode_method']}
    paths += [Path(spec['canonical_root']) / m / 'model.pt' for m in ['sparse_bf16', 'sparse_nvfp4'] if m in methods]
    assert all(p.is_file() for p in paths)
    return [{'path': str(p), 'resolved_path': str(p.resolve()), 'bytes': p.stat().st_size} for p in paths]


def inventory():
    rows = []
    for rid in ['017', '018', '022']:
        assert read(ROOT / f'runs/{rid}/results/final_audit.json')['verified']
        assert read(ROOT / f'runs/{rid}/results/worker.json')['status'] == 'complete'
    for rid in ['017', '018']:
        for path in sorted((NIGHT / 'primary_weight_cache' / rid).glob('*.safetensors')):
            record_path = ROOT / f'runs/{rid}/results/primary_weight_cache/{path.stem}.json'
            record = read(record_path)
            policy = ROOT / f'runs/{rid}/configs/policies/{path.stem}.json'
            spec = read(ROOT / f'runs/{rid}/configs/model.json')
            assert path.stat().st_size == record['bytes'] and sha(policy) == record['policy_sha256']
            command = ['/root/wja/miniconda3/envs/vllm/bin/python', str(ROOT / f'runs/{rid}/code/export_model.py'),
                '--model-path', spec['model_path'], '--policy-json', str(policy), '--output-dir', str(Path(record['source']).parent),
                '--cutlass-wrapper-path', str(REPO / 'fake/kernels/cutlass/cutlass_wrapper')]
            for method in ['sparse_bf16', 'sparse_nvfp4']:
                if method in policy.read_text():
                    command += ['--canonical-' + method.replace('_', '-') + '-state', str(Path(spec['canonical_root']) / method / 'model.pt')]
            rows.append({'path': str(path), 'source_run': rid, 'model_run': rid, 'policy': str(policy), 'policy_sha256': sha(policy),
                'known_sha256': record['sha256'], 'sha256_source': str(record_path),
                'rebuild_command': command, 'dependencies': dependencies(rid, policy)})
    for path in sorted((ROOT / 'runs/022/data/checkpoints').glob('*/*/model.safetensors')):
        rid, pid = path.parents[1].name, path.parent.name
        command_path, = (ROOT / 'runs/022/logs').glob(f'export_{rid}_{pid}_*.command.json')
        command = read(command_path)['command']
        policy = Path(command[command.index('--policy-json') + 1])
        rows.append({'path': str(path), 'source_run': '022', 'model_run': rid, 'policy': str(policy), 'policy_sha256': sha(policy),
            'known_sha256': sha(path), 'sha256_source': 'New streaming hash before cleanup',
            'rebuild_command': command, 'command_record': str(command_path), 'dependencies': dependencies(rid, policy)})
    assert len(rows) == 45
    targets = {Path(r['path']).resolve() for r in rows}
    for row in rows:
        path = Path(row['path'])
        # The exporter refuses existing directories; never suggest --force,
        # which would erase the metadata the user explicitly wants retained.
        row['original_export_command'] = row['rebuild_command'][:]
        row['rebuild_command'][row['rebuild_command'].index('--output-dir') + 1] = str(
            HERE / 'rebuild' / row['source_run'] / row['model_run'] / Path(row['policy']).stem)
        row['restore_note'] = 'Export into the new empty directory, then copy only model.safetensors to this original path; retain existing metadata and do not use --force.'
        assert path.is_file() and not path.is_symlink() and path.suffix == '.safetensors'
        details = fingerprint(path)
        assert details['links'] == 1 and details['bytes'] > 1024**3
        with path.open('rb') as handle:
            length = int.from_bytes(handle.read(8), 'little')
            assert 0 < length < 16 * 1024**2
            header = handle.read(length)
        json.loads(header)
        row.update(details, header_sha256=hashlib.sha256(header).hexdigest())
        assert not any(Path(d['resolved_path']) in targets for d in row['dependencies'])
    return rows


def protection_snapshot(targets):
    result = {}
    for root in [ROOT, REPO / 'dev/plans', REPO / 'dev/impls']:
        for directory, subdirs, files in os.walk(root, followlinks=False):
            subdirs[:] = [d for d in subdirs if not Path(directory, d).is_symlink() and Path(directory, d) != HERE]
            for name in files:
                path = Path(directory, name)
                if path in targets:
                    continue
                mode = path.lstat().st_mode
                if stat.S_ISLNK(mode):
                    value = {'symlink': str(path.readlink())}
                elif stat.S_ISREG(mode):
                    value = {'bytes': path.stat().st_size}
                    if path.suffix in ['.safetensors', '.pt', '.bin', '.so', '.arrow']:
                        value.update(fingerprint(path))
                    else:
                        value['sha256'] = sha(path)
                else:
                    continue
                result[str(path)] = value
    return result


def process_references(rows):
    identities = {(r['device'], r['inode']) for r in rows}
    references = []
    unreadable = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            for descriptor in (process / 'fd').iterdir():
                try:
                    s = descriptor.stat()
                    if (s.st_dev, s.st_ino) in identities:
                        references.append({'pid': int(process.name), 'kind': 'fd', 'path': str(descriptor.readlink())})
                except FileNotFoundError:
                    pass
            for line in (process / 'maps').read_text().splitlines():
                fields = line.split(maxsplit=5)
                if len(fields) < 6 or not int(fields[4]):
                    continue
                major, minor = [int(x, 16) for x in fields[3].split(':')]
                if (os.makedev(major, minor), int(fields[4])) in identities:
                    references.append({'pid': int(process.name), 'kind': 'mmap', 'path': fields[5]})
        except (FileNotFoundError, ProcessLookupError):
            pass
        except PermissionError:
            unreadable.append(int(process.name))
    return {'references': references, 'unreadable_processes': unreadable}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    assert not (HERE / 'completed.json').exists()
    # These are the same locks held by export and timing workers; no processes
    # are stopped and no GPU configuration is changed.
    handles = []
    try:
        for path in [NIGHT / 'measurement.lock', NIGHT / 'export.lock', NIGHT / 'primary_weight_cache/allocation.lock']:
            handle = path.open('a')
            fcntl.flock(handle, fcntl.LOCK_EX)
            handles.append(handle)
        rows = inventory()
        usage = os.statvfs(REPO)
        manifest = {'created_at_unix': time.time(), 'targets': rows,
            'bytes': sum(r['bytes'] for r in rows), 'allocated_bytes': sum(r['allocated_bytes'] for r in rows),
            'free_before_bytes': usage.f_bavail * usage.f_frsize,
            'scope': 'Only45 reproducible completed Llama exported weight binaries; all source models, canonical states, Qwen checkpoints, code and result files retained.'}
        write(HERE / 'manifest.json', manifest)
        print('Frozen', len(rows), 'weight binaries,', round(manifest['bytes'] / 1024**3, 3), 'GiB', flush=True)
        process = process_references(rows)
        write(HERE / 'process_check.json', process)
        assert not process['references'] and not process['unreadable_processes'], process
        if not args.apply:
            print('Dry-run only; no weight deleted.', flush=True)
            return
        targets = {Path(r['path']) for r in rows}
        protected = protection_snapshot(targets)
        write(HERE / 'protected_before.json', protected)
        # Repeat the live-reference check immediately before the first unlink.
        process = process_references(rows)
        assert not process['references'] and not process['unreadable_processes'], process
        with (HERE / 'deleted.jsonl').open('x') as journal:
            for row in rows:
                path = Path(row['path'])
                assert fingerprint(path) == {k: row[k] for k in fingerprint(path)}
                path.unlink()
                journal.write(json.dumps({'path': str(path), 'bytes': row['bytes'], 'deleted_at_unix': time.time()}) + '\n')
                journal.flush()
        after = protection_snapshot(targets)
        missing = sorted(set(protected) - set(after))
        changed = [p for p in protected if p in after and protected[p] != after[p]]
        usage = os.statvfs(REPO)
        audit = {'complete': True, 'deleted_files': len(rows), 'deleted_bytes': manifest['bytes'],
            'allocated_bytes': manifest['allocated_bytes'], 'protected_files': len(protected),
            'missing_protected_files': missing, 'changed_protected_files': changed,
            'free_before_bytes': manifest['free_before_bytes'], 'free_after_bytes': usage.f_bavail * usage.f_frsize,
            'manifest_sha256': sha(HERE / 'manifest.json'), 'deleted_journal_sha256': sha(HERE / 'deleted.jsonl'),
            'remaining_target_files': [r['path'] for r in rows if Path(r['path']).exists()]}
        write(HERE / 'completed.json', audit)
        assert not missing and not changed and not audit['remaining_target_files'], audit
        print('Deleted', len(rows), 'reproducible weights; protected', len(protected), 'files unchanged.', flush=True)
    finally:
        for handle in reversed(handles):
            handle.close()


if __name__ == '__main__':
    main()
